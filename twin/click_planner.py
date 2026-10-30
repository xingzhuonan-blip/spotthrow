"""Click on the floor -> the throw that lands there.

  python run_twin.py plan

Window (PyBullet):
  * green outline   floor area the throw can reach (with the current speed limits)
  * LEFT CLICK      inside it: target. The plan is solved with the fast model
                    at once, then checked in the physics simulation (and
                    corrected) in the background; the arm then previews the swing.
  * e               export the plan (JSON) for:  run_twin.py live --plan <file>
  * q               quit
Camera: Ctrl + left drag rotates, Ctrl + middle drag pans, wheel zooms.
"""

import threading
import time
from pathlib import Path

import numpy as np
import pybullet as p

from .kinematics import ARM_JOINTS, GRASP_OFFSET, LEGS, ROOT, URDF, WRIST_LINK, place_body, root_to_body
from .planner import cubic_state, save_plan

GREEN, BLUE, RED, GREY, ORANGE = (0, .65, 0), (.1, .4, 1), (1, .1, .1), (.5, .5, .5), (1, .5, 0)
PLAN_DIR = ROOT / "twin_data" / "plans"


def mouse_ray(width, height, view_matrix, proj_matrix, mx, my):
    """World-space ray (origin, direction) through pixel (mx, my).
    Matrices as returned by PyBullet (column-major 16 floats)."""
    # No matrix inverse: the PyBullet window uses an infinite far plane, whose
    # projection matrix is singular. Only the focal terms P[0,0], P[1,1] and the
    # (rigid) view matrix are needed.
    V = np.array(view_matrix, float).reshape(4, 4).T
    P = np.array(proj_matrix, float).reshape(4, 4).T
    x = 2.0 * mx / width - 1.0
    y = 1.0 - 2.0 * my / height
    d_cam = np.array([x / P[0, 0], y / P[1, 1], -1.0])  # camera looks down -z
    R, t = V[:3, :3], V[:3, 3]
    origin = -R.T @ t                                     # camera position
    d = R.T @ d_cam
    return origin, d / np.linalg.norm(d)


def floor_hit(origin, direction, floor_z):
    if abs(direction[2]) < 1e-9:
        return None
    s = (floor_z - origin[2]) / direction[2]
    return None if s <= 0 else origin + s * direction


def open_window():
    """Open the PyBullet window. Call this BEFORE creating the ThrowPlanner:
    PyBullet only reports the camera (getDebugVisualizerCamera, needed for
    mouse picking) for the window if it is the first connection (client 0);
    otherwise it returns zeros."""
    cid = p.connect(p.GUI)
    if cid != 0:
        print(f"WARNING: PyBullet window is client {cid}, not 0: mouse picking may not work.")
    return cid


class ClickPlanner:
    def __init__(self, planner, gui=True, cid=None):
        self.pl = planner
        self.gui = gui
        self.cid = cid if cid is not None else p.connect(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.cid)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
        self.plane = p.loadURDF(str(ROOT / "plane.urdf"), (0, 0, planner.floor_z), physicsClientId=self.cid)
        self.robot = p.loadURDF(URDF, physicsClientId=self.cid)
        for i, q in enumerate(LEGS):
            p.resetJointState(self.robot, 6 + i, q, physicsClientId=self.cid)
        place_body(self.cid, self.robot, root_to_body(self.cid, self.robot), planner.body_pos, planner.body_orn)
        self.ball = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_SPHERE, radius=planner.radius,
                                                                 rgbaColor=(*ORANGE, 1), physicsClientId=self.cid),
                                      physicsClientId=self.cid)
        self._set_arm(planner.space.start_pose)
        p.resetDebugVisualizerCamera(2.6, -90, -55, (0.75, 0, planner.floor_z), physicsClientId=self.cid)
        self.plan = None           # latest finished plan
        self.preview = None        # (start, end, T, t_release, p0, v0) being animated
        self.busy = False
        self.status = "Left-click inside the green area to choose a target."
        self.plan_items = []
        self.text_items = []
        self._draw_reach()
        self._draw_text()

    # ---- drawing -------------------------------------------------------

    def _set_arm(self, q):
        for j, qj in zip(ARM_JOINTS, q):
            p.resetJointState(self.robot, j, qj, physicsClientId=self.cid)

    def _grip(self):
        pos, orn = p.getLinkState(self.robot, WRIST_LINK, computeForwardKinematics=True,
                                  physicsClientId=self.cid)[4:6]
        return np.array(p.multiplyTransforms(pos, orn, GRASP_OFFSET, (0, 0, 0, 1))[0])

    def _line(self, a, b, color, width=2, store=None):
        i = p.addUserDebugLine(list(a), list(b), color, width, physicsClientId=self.cid)
        if store is not None:
            store.append(i)

    def _cross(self, pos, color, size=0.05, store=None):
        x, y, z = pos
        self._line((x - size, y, z), (x + size, y, z), color, 4, store)
        self._line((x, y - size, z), (x, y + size, z), color, 4, store)

    def _draw_reach(self):
        (rmin, rmax), (a0, a1) = self.pl.reach()
        phis = self.pl.table[:, 5]
        lo_phi, hi_phi = phis.min() + a0, phis.max() + a1
        z = self.pl.floor_z + 0.002
        c = self.pl.axis_xy
        arc = lambda r: [(c[0] + r * np.cos(f), c[1] + r * np.sin(f), z)  # noqa: E731
                         for f in np.linspace(lo_phi, hi_phi, 40)]
        for r in (rmin, rmax):
            pts = arc(r)
            for a, b in zip(pts[:-1], pts[1:]):
                self._line(a, b, GREEN, 3)
        for f in (lo_phi, hi_phi):
            self._line((c[0] + rmin * np.cos(f), c[1] + rmin * np.sin(f), z),
                       (c[0] + rmax * np.cos(f), c[1] + rmax * np.sin(f), z), GREEN, 3)

    def _draw_text(self):
        for i in self.text_items:
            p.removeUserDebugItem(i, physicsClientId=self.cid)
        lines = [self.status]
        pl = self.plan
        if pl is not None and pl["ok"]:
            P = pl["params"]
            deg = lambda q: " ".join(f"{np.degrees(v):6.1f}" for v in q)  # noqa: E731
            rel = pl["release"]
            v0 = rel["v0"]
            lines += [
                f"target ({pl['target'][0]:.2f}, {pl['target'][1]:.2f}) m",
                f"swing T = {P['duration']:.2f} s, gripper opens at {P['release_fraction']:.2f} T "
                f"(+{self.pl.gripper_delay * 1000:.0f} ms delay)",
                f"start joints [deg] sh0 sh1 el0 el1 wr0 wr1: {deg(pl['start_pose'])}",
                f"end   joints [deg] sh0 sh1 el0 el1 wr0 wr1: {deg(pl['end_pose'])}",
                f"peak joint speed {self.pl.peak_speed(pl['x']):.2f} rad/s (limit {self.pl.space.max_vel})",
                f"release speed {np.linalg.norm(v0):.2f} m/s, elevation "
                f"{np.degrees(np.arctan2(v0[2], np.linalg.norm(v0[:2]))):.0f} deg",
            ]
            if pl.get("sim_landing") is not None:
                err = np.linalg.norm(pl["sim_landing"] - pl["target"]) * 100
                lines.append(f"physics check: lands ({pl['sim_landing'][0]:.3f}, {pl['sim_landing'][1]:.3f}), "
                             f"{err:.1f} cm from target")
            lines.append("e = export plan for the robot, q = quit")
        self.text_items = [p.addUserDebugText(t, (0.2, 1.3, 0.9 - 0.08 * i), (0, 0, 0), 1.1,
                                              physicsClientId=self.cid) for i, t in enumerate(lines)]

    def _draw_plan(self, plan, physics_done):
        for i in self.plan_items:
            p.removeUserDebugItem(i, physicsClientId=self.cid)
        self.plan_items = []
        tz = self.pl.floor_z + 0.003
        self._cross((*plan["target"], tz), GREEN, 0.08, self.plan_items)
        rel = plan["release"]
        # Predicted ball path from the release state (no drag).
        t = np.linspace(0, rel.get("t_flight", 0.0), 30)
        path = rel["p0"] + np.outer(t, rel["v0"]) + np.outer(0.5 * t**2, (0, 0, -9.81))
        path[:, :2] += plan["residual"]  # include the physics correction
        for a, b in zip(path[:-1], path[1:]):
            self._line(a, b, BLUE, 2, self.plan_items)
        # Gripper path over the swing.
        start, end, T = plan["start_pose"], plan["end_pose"], plan["params"]["duration"]
        pts = []
        for s in np.linspace(0, T, 25):
            self._set_arm(cubic_state(start, end, T, s))
            pts.append(self._grip())
        for a, b in zip(pts[:-1], pts[1:]):
            self._line(a, b, GREY, 2, self.plan_items)
        if physics_done and plan.get("sim_landing") is not None:
            self._cross((*plan["sim_landing"], tz), RED, 0.05, self.plan_items)
        self.preview = (start, end, T, rel["t_release"], rel["p0"], rel["v0"], plan["residual"])
        self.preview_t0 = time.time()

    # ---- planning (background thread) ---------------------------------

    def click(self, target_xy):
        if self.busy:
            return
        self.busy = True
        self.status = f"Planning for ({target_xy[0]:.2f}, {target_xy[1]:.2f})..."
        self._draw_text()
        threading.Thread(target=self._plan_worker, args=(np.array(target_xy),), daemon=True).start()

    def _plan_worker(self, target):
        try:
            self.pl.residual = np.zeros(2)
            x, land, miss = self.pl.solve(target)
            if x is None or miss > 0.05:
                self.pending = ("fail", "Out of reach: click inside the green area.")
                return
            quick = self._as_plan(x, target)
            self.pending = ("quick", quick)
            full = self.pl.plan(target)
            self.pending = ("full", full) if full["ok"] else ("fail", full["reason"])
        except Exception as err:  # keep the window alive
            self.pending = ("fail", f"error: {err}")
        finally:
            self.busy = False

    pending = None

    def _as_plan(self, x, target):
        start, end, P = self.pl._poses(x)
        return {"ok": True, "x": x, "params": P, "start_pose": start, "end_pose": end,
                "release": self.pl.predict(x), "residual": self.pl.residual.copy(), "target": target,
                "sim_landing": None}

    def _consume_pending(self):
        if self.pending is None:
            return
        kind, data = self.pending
        self.pending = None
        if kind == "fail":
            self.status = data
        elif kind == "quick":
            self.plan = data
            self.status = "Fast model plan shown; checking it in the physics simulation..."
            self._draw_plan(data, physics_done=False)
        else:
            self.plan = data
            self.status = "Plan checked in physics (red X = simulated landing)."
            self._draw_plan(data, physics_done=True)
        self._draw_text()

    # ---- animation + input ---------------------------------------------

    def _animate(self):
        if self.preview is None:
            return
        start, end, T, t_rel, p0, v0, residual = self.preview
        slow = 0.5
        t = ((time.time() - self.preview_t0) * slow) % (T + 1.5)
        self._set_arm(cubic_state(start, end, T, t))
        if t < t_rel:
            ball = self._grip()
        else:
            s = t - t_rel
            ball = p0 + v0 * s + np.array([0, 0, -4.905 * s * s])
            ball[:2] += residual * min(1.0, s / max(self.plan["release"].get("t_flight", 1.0), 1e-3))
            ball[2] = max(ball[2], self.pl.floor_z + self.pl.radius)
        p.resetBasePositionAndOrientation(self.ball, ball, (0, 0, 0, 1), physicsClientId=self.cid)

    def export(self):
        if self.plan is None or not self.plan.get("ok"):
            self.status = "Nothing to export yet."
            return None
        PLAN_DIR.mkdir(parents=True, exist_ok=True)
        path = PLAN_DIR / f"plan_{time.strftime('%Y%m%d_%H%M%S')}.json"
        save_plan(self.plan, path)
        self.status = f"Exported {path.name}"
        print(f"exported {path}\n  simulate:  python run_twin.py simulate --plan {path}\n"
              f"  on Spot:   python run_twin.py live 10.0.0.30 --local-ip <this PC> --plan {path} --live-view")
        return path

    def run(self):
        while p.isConnected(self.cid):
            for ev in p.getMouseEvents(physicsClientId=self.cid):
                etype, mx, my, button, state = ev
                if etype == 2 and button == 0 and state & p.KEY_WAS_RELEASED:
                    w, h, view, proj = p.getDebugVisualizerCamera(physicsClientId=self.cid)[:4]
                    if w == 0 or h == 0:
                        self.status = "PyBullet gave no camera info: open the window first (see open_window)."
                        self._draw_text()
                        continue
                    hit = floor_hit(*mouse_ray(w, h, view, proj, mx, my), self.pl.floor_z)
                    if hit is not None:
                        self.click(hit[:2])
            keys = p.getKeyboardEvents(physicsClientId=self.cid)
            pressed = {k for k, s in keys.items() if s & p.KEY_WAS_TRIGGERED}
            if ord("q") in pressed:
                break
            if ord("e") in pressed:
                self.export()
                self._draw_text()
            self._consume_pending()
            self._animate()
            time.sleep(1 / 60)
        if p.isConnected(self.cid):
            p.disconnect(self.cid)
