"""A real throw: wind up, accelerate, hold top speed while the gripper lets go,
then brake.

The original ThrowSpace moves start -> end with one cubic, which is at rest at
both ends, so the arm is already slowing down when the ball leaves, and the
gripper needs ~0.1 s to let go. That gives slow "drops" (~0.4 m/s).

Here every moving joint (sh1, el0, wr0; sh0 aims) follows the same timing:

    accelerate (ta)  ->  coast at constant speed (tc)  ->  brake (tb = ta)

with smooth (cosine) speed ramps. The gripper opens during the coast and the
ball is meant to be free right when braking starts (release_at="brake"): if
it lets go a bit earlier it still leaves at full speed, a bit later it leaves
while the gripper slows down, so the gripper ends up behind the ball and
cannot hit it again. A ball freed while the arm still accelerates falls
behind the gripper and is hit when the gripper brakes (seen on Spot,
2026-11-03); rehit() checks this over a range of real gripper delays.
Old plans (release_at="mid") freed the ball in the middle of the coast.

Parameters x = (sh1_r, el0_r, wr0_r, w_sh1, w_el0, w_wr0, sh0):
release pose of the three pitch joints, their speeds during the coast
(rad/s), and the shoulder yaw (aim). Joint speed and acceleration limits fix
the ramp times.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from throw_trajectory import JOINT_LIMIT_MARGIN, JOINT_LIMITS, check_trajectory

G = 9.81
EL1, WR1 = 0.0, 1.57  # unused roll joints, as in the other throws


@dataclass
class FastThrowSpace:
    max_vel: float = 2.5      # rad/s, per joint (lab example; hard cap in spot_throw_safe is 8)
    max_acc: float = 10.0     # rad/s^2
    coast: float = 0.24       # s at constant speed around the release
    gripper_delay: float = 0.1
    min_ramp: float = 0.15    # s, gentlest ramp allowed
    clearance: float = 0.03   # m, arm links vs body along the whole path
    min_release_height: float = 1.0  # m above the floor: overhand, and high enough for OptiTrack
    release_at: str = "brake"  # "brake": ball free when braking starts; "mid": middle of the coast (old plans)
    open_margin: float = 0.05  # s, open command at least this long after the coast starts

    names = ("sh1_r", "el0_r", "wr0_r", "w_sh1", "w_el0", "w_wr0", "sh0")

    def timing(self, w):
        """Ramp time from the fastest joint and the acceleration limit
        (cosine ramp: peak acceleration = pi * w / (2 * ta))."""
        wmax = float(np.max(np.abs(w)))
        return max(self.min_ramp, np.pi * wmax / (2 * self.max_acc))

    def coast_time(self):
        """With release_at="brake" the open command (gripper_delay before the
        brake) must fall inside the coast, never while the arm accelerates."""
        if self.release_at == "brake":
            return self.gripper_delay + self.open_margin  # short coast = short wind-up (body clearance)
        return self.coast

    def free_time(self, ta, tc):
        return ta + tc if self.release_at == "brake" else ta + tc / 2

    def build(self, x, dt=0.01):
        """x -> (duration, release_fraction, (times, positions, velocities)),
        the same trajectory format as the other throws."""
        sh1_r, el0_r, wr0_r, w1, w2, w3, sh0 = x
        w = np.array([0.0, w1, w2, 0.0, w3, 0.0])
        q_r = np.array([sh0, sh1_r, el0_r, EL1, wr0_r, WR1])
        ta = self.timing(w)
        tc = self.coast_time()
        T = 2 * ta + tc
        t_free = self.free_time(ta, tc)
        q0 = q_r - w * (ta / 2 + (t_free - ta))
        times = np.linspace(0.0, T, int(round(T / dt)) + 1)
        s = np.zeros_like(times)        # speed fraction 0..1
        d = np.zeros_like(times)        # distance travelled, in units of "full speed seconds"
        for i, t in enumerate(times):
            if t < ta:
                s[i] = 0.5 * (1 - np.cos(np.pi * t / ta))
                d[i] = 0.5 * (t - ta / np.pi * np.sin(np.pi * t / ta))
            elif t < ta + tc:
                s[i] = 1.0
                d[i] = ta / 2 + (t - ta)
            else:
                u = t - ta - tc
                s[i] = 0.5 * (1 + np.cos(np.pi * u / ta))
                d[i] = ta / 2 + tc + 0.5 * (u + ta / np.pi * np.sin(np.pi * u / ta))
        positions = q0 + np.outer(d, w)
        velocities = np.outer(s, w)
        release_fraction = (t_free - self.gripper_delay) / T
        return T, release_fraction, (times, positions, velocities)

    def check(self, x):
        """Joint limits, speed and acceleration (same check as the robot uses)."""
        T, rf, traj = self.build(x)
        return check_trajectory(*traj, self.max_vel, self.max_acc)


class FastThrowPlanner:
    """Plans fast throws. Reuses ThrowPlanner for the standing pose, forward
    kinematics and the physics simulation."""

    def __init__(self, space=None, base=None, ball="tennis"):
        from .planner import ThrowPlanner
        self.space = space or FastThrowSpace()
        self.base = base or ThrowPlanner(gripper_delay=self.space.gripper_delay, ball=ball)

    # ---- forward model ----
    def release_state(self, x):
        T, rf, (times, pos, vel) = self.space.build(x)
        t_free = rf * T + self.space.gripper_delay
        q = np.array([np.interp(t_free, times, pos[:, j]) for j in range(6)])
        v = np.array([np.interp(t_free, times, vel[:, j]) for j in range(6)])
        h = 1e-3
        p0 = self.base._grip(q)
        v0 = (self.base._grip(q + v * h) - self.base._grip(q - v * h)) / (2 * h)
        return p0, v0

    def predict(self, x):
        """Release position/velocity and landing (no drag), robot frame."""
        p0, v0 = self.release_state(x)
        ground = self.base.floor_z + self.base.radius
        a, b, c = -0.5 * G, v0[2], p0[2] - ground
        disc = b * b - 4 * a * c
        if disc < 0:
            return {"p0": p0, "v0": v0, "landing": None}
        tf = (-b - np.sqrt(disc)) / (2 * a)
        return {"p0": p0, "v0": v0, "landing": p0[:2] + v0[:2] * tf, "t_flight": tf}

    max_rehit = 0.01  # m: the ball may not come back closer than this to the gripper

    STOW = np.array([0.0, -3.11, 3.13, 0.0, 1.57, 0.0])  # approximate

    def min_clearance(self, x, n=40):
        """Smallest arm-to-body distance along the throw AND along the slow
        move from stow to the wind-up pose (robot: initial_move), sampled.
        The stowed pose itself rests on the body, so that move is checked
        from 15 % of the way."""
        import pybullet as p
        from .kinematics import ARM_JOINTS
        fk = self.base.fk
        T, rf, (times, pos, vel) = self.space.build(x)
        approach = [self.STOW + (pos[0] - self.STOW) * s for s in np.linspace(0.15, 1.0, 20)]
        best = 1.0
        for q in list(pos[np.linspace(0, len(times) - 1, n).astype(int)]) + approach:
            for j, qj in zip(ARM_JOINTS, q):
                p.resetJointState(fk.robot, j, qj, physicsClientId=fk.cid)
            for a in (21, 22, 23, 24, 25):
                for b in range(2, 18):
                    for c in p.getClosestPoints(fk.robot, fk.robot, 0.1, linkIndexA=a, linkIndexB=b,
                                                physicsClientId=fk.cid):
                        best = min(best, c[8])
        return best

    def rehit(self, x, delays=None, after=0.6, dt=0.005):
        """Does the gripper come back at the ball after it lets go?
        For each real gripper delay in `delays` (default 0.05 s .. planned +
        0.15 s) the ball flies from the gripper (gravity only) while the
        gripper follows the trajectory and then holds still. Returns the worst
        "approach" in m: how much the ball-gripper distance shrinks again after
        growing. 0 = they only move apart. inf = the ball would still be held
        when the arm stops."""
        from .kinematics import ARM_JOINTS  # noqa: F401  (fk via base._grip)
        T, rf, (times, pos, vel) = self.space.build(x)
        t_open = rf * T
        if delays is None:
            delays = np.linspace(0.05, self.space.gripper_delay + 0.15, 6)
        q_at = lambda t: np.array([np.interp(min(t, T), times, pos[:, j]) for j in range(6)])
        grid = np.arange(t_open + min(delays), min(t_open + max(delays), T) + after, dt)
        grip = np.array([self.base._grip(q_at(t)) for t in grid])
        worst = 0.0
        g = np.array([0.0, 0.0, -G])
        for d in delays:
            tr = t_open + d
            if tr >= T:
                return float("inf")
            i = int(np.searchsorted(grid, tr))
            if i + 2 >= len(grid):
                return float("inf")
            p0 = grip[i]
            v0 = (grip[i + 1] - grip[i - 1]) / (2 * dt) if i > 0 else (grip[i + 1] - grip[i]) / dt
            tt = grid[i:] - grid[i]
            ball = p0 + np.outer(tt, v0) + 0.5 * np.outer(tt**2, g)
            dist = np.linalg.norm(ball - grip[i:], axis=1)
            dist = dist[ball[:, 2] > self.base.floor_z + self.base.radius]  # until it lands
            if len(dist):
                worst = max(worst, float(np.max(np.maximum.accumulate(dist) - dist)))
        return worst

    # ---- search ----
    def _bounds(self):
        m = self.space.max_vel
        lo = JOINT_LIMITS[:, 0] + JOINT_LIMIT_MARGIN
        hi = JOINT_LIMITS[:, 1] - JOINT_LIMIT_MARGIN
        return [(lo[1], hi[1]), (lo[2], hi[2]), (lo[4], hi[4]),
                (0.0, m), (-m, 0.0), (-m, m), (-0.9, 0.9)]

    def _penalty(self, x):
        problems = self.space.check(x)
        pen = 0.0
        if problems:
            T, rf, (t, pos, vel) = self.space.build(x)
            lo = JOINT_LIMITS[:, 0] + JOINT_LIMIT_MARGIN
            hi = JOINT_LIMITS[:, 1] - JOINT_LIMIT_MARGIN
            pen += np.sum(np.clip(lo - pos.min(0), 0, None)) + np.sum(np.clip(pos.max(0) - hi, 0, None))
            pen += np.sum(np.clip(np.abs(vel).max(0) - self.space.max_vel, 0, None))
        return pen

    def solve(self, target_xy, n_starts=40, seed=0, min_release_speed=1.5, aim_xy=None):
        """Hit target_xy (robot frame, m). Among hits, prefer lower joint speeds
        and a release angle near 40 deg (least sensitive to speed errors).
        min_release_speed (m/s) keeps it a throw: without it, short targets
        become "reach out and drop". aim_xy: point the model should hit
        (target shifted by the physics correction); defaults to target_xy."""
        target_xy = np.asarray(target_xy if aim_xy is None else aim_xy, float)
        rng = np.random.default_rng(seed)
        bounds = self._bounds()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        def cost(x):
            x = np.clip(x, lo, hi)
            pen = self._penalty(x)
            pr = self.predict(x)
            if pr["landing"] is None:
                return 100 + pen * 100
            miss = np.linalg.norm(pr["landing"] - target_xy)
            v0 = pr["v0"]
            elev = np.degrees(np.arctan2(v0[2], np.linalg.norm(v0[:2])))
            speed = np.max(np.abs(x[3:6])) / self.space.max_vel
            slow = max(0.0, min_release_speed - np.linalg.norm(v0))
            low = max(0.0, self.space.min_release_height - (pr["p0"][2] - self.base.floor_z))
            return 20 * low**2 + 50 * miss**2 + 0.02 * speed + 0.0005 * (elev - 40) ** 2 + 100 * pen + 10 * slow**2

        best = None
        for _ in range(n_starts):
            x0 = lo + rng.random(len(lo)) * (hi - lo)
            x0[6] = np.arctan2(target_xy[1] - self.base.axis_xy[1], target_xy[0] - self.base.axis_xy[0])
            r = minimize(cost, x0, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-7})
            x = np.clip(r.x, lo, hi)
            if self._penalty(x) > 0:
                continue
            pr = self.predict(x)
            if pr["landing"] is None:
                continue
            miss = np.linalg.norm(pr["landing"] - target_xy)
            if miss > 0.03 or np.linalg.norm(pr["v0"]) < min_release_speed - 0.05:
                continue
            if pr["p0"][2] - self.base.floor_z < self.space.min_release_height:
                continue
            if self.min_clearance(x) < self.space.clearance:
                continue
            if self.rehit(x) > self.max_rehit:
                continue
            if best is None or r.fun < best[1]:
                best = (x, r.fun, miss)
        return None if best is None else best[0]

    def simulate(self, x):
        """Full physics check: landing in the robot frame (None if no landing)."""
        from .geometry import inv_T, transform_points
        from .sources import world_log_to_robot_frame
        T, rf, traj = self.space.build(x)
        res = self.base.sim.simulate(T, rf, trajectory=traj)
        if res["landing"] is None:
            return None, res
        T_world_robot = world_log_to_robot_frame(res["log"])[0]
        return transform_points(inv_T(T_world_robot), res["landing"][1][None])[0], res

    def plan(self, target_xy, min_release_speed=1.5, sim_iters=3, tol=0.02, n_starts=24, verbose=False):
        """Solve, check in physics (drag, gripper), shift the aim point by the
        error, re-solve. Returns a dict for save_fast_plan, or None."""
        target_xy = np.asarray(target_xy, float)
        aim = target_xy.copy()
        x = sim_land = None
        for k in range(sim_iters):
            x = self.solve(target_xy, n_starts=n_starts, min_release_speed=min_release_speed, aim_xy=aim)
            if x is None:
                return None
            sim_land, _ = self.simulate(x)
            if sim_land is None:
                return None
            err = sim_land[:2] - target_xy
            if verbose:
                print(f"  physics check {k + 1}: lands {np.round(sim_land[:2], 3)}, error {np.linalg.norm(err) * 100:.1f} cm")
            if np.linalg.norm(err) < tol:
                break
            aim = aim - err
        return {"x": np.asarray(x).tolist(), "target": target_xy.tolist(),
                "sim_landing": sim_land[:2].tolist(), **self.describe(x)}

    def describe(self, x):
        T, rf, (t, pos, vel) = self.space.build(x)
        pr = self.predict(x)
        v0 = pr["v0"]
        return {"duration": T, "release_fraction": rf, "start_pose": pos[0].tolist(),
                "end_pose": pos[-1].tolist(), "release_speed": float(np.linalg.norm(v0)),
                "elevation_deg": float(np.degrees(np.arctan2(v0[2], np.linalg.norm(v0[:2])))),
                "release_height": float(pr["p0"][2] - self.base.floor_z),
                "peak_joint_speed": float(np.abs(vel).max()),
                "predicted_landing": None if pr["landing"] is None else pr["landing"].tolist(),
                "clearance": self.min_clearance(x), "rehit": self.rehit(x)}

    def close(self):
        self.base.close()


def save_fast_plan(plan, space, path):
    """Same file format family as planner.save_plan; load_plan reads both."""
    import json
    from dataclasses import asdict
    from pathlib import Path
    d = {"kind": "fast", "space": asdict(space), **plan}
    Path(path).write_text(json.dumps(d, indent=2, default=float))
