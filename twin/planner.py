"""Target -> throw: compute the trajectory (joint angles, timing) that lands
the ball on a given floor point.

Forward model (fast, no physics): the ball leaves the gripper at
release_fraction * T + gripper_delay with the gripper's velocity, from
forward kinematics of the commanded trajectory; then free flight. One
evaluation takes well under a millisecond, so we can search.

Inverse: shoulder yaw sh0 rotates the whole arm about a vertical axis, so the
aim is (almost) analytic: search duration / release / shoulder range for the
right DISTANCE from that axis, then turn sh0 to the right DIRECTION, then
polish all four parameters together. Among solutions the gentlest one (lowest
peak joint speed) wins.

Correction: the forward model ignores tracking error and body sway. A full
physics simulation of the plan measures the landing error, the target is
shifted by it and the plan recomputed (the same idea will use real throws).
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from throw_trajectory import JOINT_LIMIT_MARGIN, JOINT_LIMITS, check_trajectory, cubic_joint_trajectory

from .geometry import flatten, inv_T, make_T, T_to_pos_quat, transform_points
from .kinematics import ArmFK
from .optimizer import ThrowSpace

G = 9.81


def cubic_state(start, end, T, t):
    s = np.clip(t / T, 0.0, 1.0)
    d = np.asarray(end) - np.asarray(start)
    return np.asarray(start) + (3 * s**2 - 2 * s**3) * d


class ThrowPlanner:
    def __init__(self, space=None, gripper_delay=0.1, ball="tennis", sim_source=None):
        from .sources import SimSource
        self.space = space or ThrowSpace(upper=(2.0, 0.8, -0.6, 0.9), lower=(0.9, 0.3, -1.6, -0.9))
        self.gripper_delay = gripper_delay
        self.sim = sim_source or SimSource(ball=ball, gripper_delay=gripper_delay)
        from sim_spot_throwing import Spot
        self.radius = Spot.ball_parameters[ball].radius
        # Standing pose from a short simulation: body tilt and floor height in
        # the robot frame (same frame as the records).
        start, _, _ = self._poses(np.mean(self.space.bounds(), axis=0))
        still = (np.array([0.0, 0.3]), np.array([start, start]), np.zeros((2, 6)))
        log = self.sim.simulate(0.3, 0.99, trajectory=still)["log"]
        T_world_body = make_T(log["body_pos"][0], log["body_orn"][0])
        T_world_robot = flatten(T_world_body)
        self.body_pos, self.body_orn = T_to_pos_quat(inv_T(T_world_robot) @ T_world_body)
        self.floor_z = -T_world_robot[2, 3]
        self.fk = ArmFK()
        self.axis_xy = self._sh0_axis()
        self.table = None
        self.residual = np.zeros(2)  # learned correction (sim or real throws)

    # ---- forward model -------------------------------------------------

    def _poses(self, x):
        p = dict(zip(self.space.names, x))
        start = list(self.space.start_pose)
        start[0] = p["sh0"]
        end = [p["sh0"], p["sh1_end"], self.space.el0_end, start[3], start[4], start[5]]
        return start, end, p

    def _grip(self, q):
        return self.fk.grasp_point(q, self.body_pos, self.body_orn)

    def _sh0_axis(self):
        """Where the shoulder-yaw axis meets the floor plane (xy)."""
        q = list(self.space.start_pose)
        pts = []
        for a in (-0.4, 0.0, 0.4):
            q[0] = a
            pts.append(self._grip(q)[:2])
        (x1, y1), (x2, y2), (x3, y3) = pts  # circle through three points
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
        uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
        return np.array([ux, uy])

    def predict(self, x):
        """Release state and landing point (no drag) of parameter vector x."""
        start, end, p = self._poses(x)
        T = p["duration"]
        t_r = min(p["release_fraction"] * T + self.gripper_delay, T)
        h = 1e-3
        p0 = self._grip(cubic_state(start, end, T, t_r))
        v0 = (self._grip(cubic_state(start, end, T, t_r + h)) - self._grip(cubic_state(start, end, T, t_r - h))) / (2 * h)
        ground = self.floor_z + self.radius
        a, b, c = -0.5 * G, v0[2], p0[2] - ground
        disc = b * b - 4 * a * c
        if disc < 0:
            return {"p0": p0, "v0": v0, "landing": None, "t_release": t_r}
        tf = (-b - np.sqrt(disc)) / (2 * a)
        landing = np.array([p0[0] + v0[0] * tf, p0[1] + v0[1] * tf, ground])
        return {"p0": p0, "v0": v0, "landing": landing, "t_release": t_r, "t_flight": tf}

    def peak_speed(self, x):
        start, end, p = self._poses(x)
        return 1.5 * np.max(np.abs(np.subtract(end, start))) / p["duration"]

    def feasible(self, x):
        """Same check as throw_trajectory.check_trajectory, in closed form for
        a cubic (peak speed 1.5*d/T, peak acceleration 6*d/T^2, monotonic
        motion so only the end points can hit a joint limit). ~1000x faster."""
        start, end, p = self._poses(x)
        d = np.abs(np.subtract(end, start))
        T = p["duration"]
        if np.max(1.5 * d / T) > self.space.max_vel or np.max(6 * d / T**2) > self.space.max_acc:
            return False
        lim_lo = JOINT_LIMITS[:, 0] + JOINT_LIMIT_MARGIN
        lim_hi = JOINT_LIMITS[:, 1] - JOINT_LIMIT_MARGIN
        return bool(np.all((np.array([start, end]) >= lim_lo) & (np.array([start, end]) <= lim_hi)))

    # ---- reachable set -------------------------------------------------

    STOW = np.array([0.0, -3.11, 3.13, 0.0, 1.57, 0.0])  # approximate

    def min_clearance(self, x, n=25):
        """Smallest arm-to-body distance (m) along the swing, start pose
        included. Needed once the aim can turn the arm over Spot's sides/back.
        The slow move from stow is not checked here: with the approximate stow
        pose the simulated wrist grazes the body (-6 cm) on a move Spot has
        made safely many times."""
        import pybullet as pb
        from .kinematics import ARM_JOINTS
        start, end, P = self._poses(x)
        T = P["duration"]
        qs = [cubic_state(start, end, T, t) for t in np.linspace(0, T, n)]
        best = 1.0
        for q in qs:
            for j, qj in zip(ARM_JOINTS, q):
                pb.resetJointState(self.fk.robot, j, qj, physicsClientId=self.fk.cid)
            for a in (21, 22, 23, 24, 25):
                for b in range(2, 18):
                    for c in pb.getClosestPoints(self.fk.robot, self.fk.robot, 0.1, linkIndexA=a, linkIndexB=b,
                                                 physicsClientId=self.fk.cid):
                        best = min(best, c[8])
        return best

    def build_table(self, n=(10, 9, 9)):
        """Landings for sh0 = 0 over a grid of (duration, release, sh1_end)."""
        lo, hi = self.space.bounds()
        rows = []
        for T in np.linspace(lo[0], hi[0], n[0]):
            for rf in np.linspace(lo[1], hi[1], n[1]):
                for s1 in np.linspace(lo[2], hi[2], n[2]):
                    x = np.array([T, rf, s1, 0.0])
                    if not self.feasible(x):
                        continue
                    pr = self.predict(x)
                    if pr["landing"] is None or pr["v0"][0] <= 0:
                        continue  # only forward throws
                    rel = pr["landing"][:2] - self.axis_xy
                    rows.append((*x, np.linalg.norm(rel), np.arctan2(rel[1], rel[0]), self.peak_speed(x)))
        self.table = np.array(rows)
        return self.table

    def reach(self):
        """(min, max) distance from the shoulder axis, and the yaw range."""
        if self.table is None:
            self.build_table()
        lo, hi = self.space.bounds()
        return (self.table[:, 4].min(), self.table[:, 4].max()), (lo[3], hi[3])

    # ---- inverse -------------------------------------------------------

    def solve(self, target_xy, n_starts=6):
        """Parameters whose predicted landing hits target_xy (robot frame).
        Returns (x, predicted landing, miss in m)."""
        if self.table is None:
            self.build_table()
        target = np.asarray(target_xy, float) - self.residual
        rel = target - self.axis_xy
        R, phi = np.linalg.norm(rel), np.arctan2(rel[1], rel[0])
        lo, hi = self.space.bounds()
        # Candidates: right distance (cheapest first), sh0 turns them onto the target.
        err = np.abs(self.table[:, 4] - R)
        order = np.lexsort((self.table[:, 6], np.round(err / 0.02)))
        starts = []
        for row in self.table[order]:
            sh0 = phi - row[5]
            if lo[3] <= sh0 <= hi[3]:
                starts.append(np.array([row[0], row[1], row[2], sh0]))
            if len(starts) >= n_starts:
                break
        if not starts:
            return None, None, np.inf

        def cost(xs):
            x = lo + xs * (hi - lo)
            if not self.feasible(x):
                return 10.0
            land = self.predict(x)["landing"]
            if land is None:
                return 10.0
            return np.sum((land[:2] - target) ** 2) + 1e-5 * self.peak_speed(x)

        best = None
        for x0 in starts:
            res = minimize(cost, (x0 - lo) / (hi - lo), method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 400})
            if best is None or res.fun < best.fun:
                best = res
        x = lo + np.clip(best.x, 0, 1) * (hi - lo)
        land = self.predict(x)["landing"]
        miss = np.inf if land is None else float(np.linalg.norm(land[:2] + self.residual - target_xy))
        return x, land, miss

    # ---- physics check + correction -----------------------------------

    def trajectory(self, x):
        start, end, p = self._poses(x)
        return p["duration"], p["release_fraction"], cubic_joint_trajectory(start, end, p["duration"])

    def simulate(self, x):
        """Full physics simulation of plan x: landing in the robot frame."""
        from .sources import world_log_to_robot_frame
        duration, rf, traj = self.trajectory(x)
        res = self.sim.simulate(duration, rf, trajectory=traj)
        if res["landing"] is None:
            return None, res
        T_world_robot = world_log_to_robot_frame(res["log"])[0]
        return transform_points(inv_T(T_world_robot), res["landing"][1][None])[0], res

    def plan(self, target_xy, sim_iters=2, tol=0.01, verbose=False):
        """Solve, check in physics, shift the target by the error, re-solve."""
        target_xy = np.asarray(target_xy, float)
        self.residual = np.zeros(2)
        x, land, miss = self.solve(target_xy)
        if x is None or miss > 0.05:
            return {"ok": False, "x": x, "reason": "out of reach"}
        sim_land = None
        for k in range(sim_iters):
            sim_land, _ = self.simulate(x)
            if sim_land is None:
                break
            err = sim_land[:2] - target_xy
            if verbose:
                print(f"  physics check {k + 1}: lands {np.round(sim_land[:2], 3)}, "
                      f"error {np.linalg.norm(err) * 100:.1f} cm")
            if np.linalg.norm(err) < tol:
                break
            # model error = physics - prediction (the prediction already includes the residual)
            self.residual += err
            x, land, _ = self.solve(target_xy)
            if x is None:
                return {"ok": False, "x": None, "reason": "out of reach after correction"}
        else:
            sim_land, _ = self.simulate(x)
        clear = self.min_clearance(x)
        if clear < 0.03:
            return {"ok": False, "x": x, "reason": f"arm would pass {clear * 100:.0f} cm from the body"}
        start, end, p = self._poses(x)
        pr = self.predict(x)
        return {"ok": True, "x": x, "params": p, "start_pose": start, "end_pose": end,
                "predicted_landing": None if land is None else land[:2] + self.residual,
                "sim_landing": None if sim_land is None else sim_land[:2],
                "release": pr, "residual": self.residual.copy(), "target": target_xy}

    def close(self):
        self.fk.close()


def save_plan(plan, path):
    p = plan["params"]
    Path(path).write_text(json.dumps({
        "target": list(map(float, plan["target"])),
        "duration": float(p["duration"]), "release_fraction": float(p["release_fraction"]),
        "start_pose": [float(v) for v in plan["start_pose"]], "end_pose": [float(v) for v in plan["end_pose"]],
        "sim_landing": None if plan["sim_landing"] is None else [float(v) for v in plan["sim_landing"]],
        "gripper_delay": float(plan.get("gripper_delay", 0.1)),
    }, indent=2))


def load_plan(path):
    d = json.loads(Path(path).read_text())
    if d.get("kind") == "fast":  # twin/fast_throw.py: windup / coast / brake profile
        from .fast_throw import FastThrowSpace
        space = {"release_at": "mid", **d["space"]}  # plans saved before 2026-11-03: mid-coast release
        T, rf, traj = FastThrowSpace(**space).build(d["x"])
        d["duration"], d["release_fraction"] = T, rf
        d["gripper_delay"] = d["space"].get("gripper_delay", 0.1)
        return d, traj
    d.setdefault("gripper_delay", 0.1)  # older plans: the planner default
    traj = cubic_joint_trajectory(d["start_pose"], d["end_pose"], d["duration"])
    return d, traj
