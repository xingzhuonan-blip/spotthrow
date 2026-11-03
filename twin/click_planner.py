"""Click on the floor -> the throw that lands there.

  python run_twin.py plan

Window (PyBullet):
  * LEFT CLICK      anywhere: target. The plan is solved with the fast model
                    at once, then checked in the physics simulation (and
                    corrected) in the background; the arm then previews the swing.
  * e               export the plan (JSON) for:  run_twin.py live --plan <file>
  * q               quit
Camera: RIGHT drag rotates, MIDDLE (wheel button) drag pans, wheel zooms.
Floor: grid every 0.5 m around Spot, blue rings
every 0.5 m labelled with the distance; the cursor's floor position and
distance are shown live at the top.
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
    def __init__(self, planner, gui=True, cid=None, fast_fallback=False):
        self.pl = planner
        self.fast_fallback = fast_fallback  # try a fast throw when the gentle one can't
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
        self.fast = None           # FastThrowPlanner, made on the first far click
        self.preview = None        # (start, end, T, t_release, p0, v0) being animated
        self.busy = False
        self.status = "Left-click a target on the floor."
        self.plan_items = []
        self.text_items = []
        self.drag = None           # (button, x, y) while the right / middle button is held
        self.cursor_item = -1
        self._last_cursor = 0.0
        self._draw_floor_grid()
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

    def _draw_floor_grid(self, step=0.5, x_range=(-1.0, 3.0), y_range=(-2.0, 2.0), rings=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
        """Distance aids in the robot frame (origin = Spot's body centre on the floor)."""
        z = self.pl.floor_z + 0.001
        light = (.75, .75, .75)
        for x in np.arange(x_range[0], x_range[1] + 1e-9, step):
            self._line((x, y_range[0], z), (x, y_range[1], z), light, 1)
        for y in np.arange(y_range[0], y_range[1] + 1e-9, step):
            self._line((x_range[0], y, z), (x_range[1], y, z), light, 1)
        for r in rings:
            f = np.linspace(-np.pi / 2, np.pi / 2, 61)  # front half: where throws go
            pts = [(r * np.cos(a), r * np.sin(a), z) for a in f]
            for a, b in zip(pts[:-1], pts[1:]):
                self._line(a, b, (.45, .55, .9), 1)
            p.addUserDebugText(f"{r:g} m", (r * 0.707 + 0.03, -r * 0.707, z), (.2, .3, .8), 1.0,
                               physicsClientId=self.cid)

    def _show_cursor(self, mx, my):
        now = time.time()
        if now - self._last_cursor < 0.08:
            return
        self._last_cursor = now
        w, h, view, proj = p.getDebugVisualizerCamera(physicsClientId=self.cid)[:4]
        if w == 0 or h == 0:
            return
        hit = floor_hit(*mouse_ray(w, h, view, proj, mx, my), self.pl.floor_z)
        if hit is None:
            return
        text = f"cursor: x {hit[0]:+.2f}  y {hit[1]:+.2f} m   distance {np.hypot(hit[0], hit[1]):.2f} m"
        self.cursor_item = p.addUserDebugText(text, (0.2, 1.3, 1.05), (0, 0, .8), 1.3,
                                              replaceItemUniqueId=self.cursor_item, physicsClientId=self.cid)

    def _camera_mouse(self, etype, mx, my, button, state):
        """Right drag = rotate, middle drag = pan (no Ctrl needed)."""
        if etype == 2 and button in (1, 2):
            if state & p.KEY_WAS_TRIGGERED:
                self.drag = (button, mx, my)
            elif state & p.KEY_WAS_RELEASED:
                self.drag = None
            return
        if etype != 1 or self.drag is None:
            return
        b, x0, y0 = self.drag
        dx, dy = mx - x0, my - y0
        self.drag = (b, mx, my)
        cam = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        horiz, vert, yaw, pitch, dist, target = (np.array(cam[6]), np.array(cam[7]), cam[8], cam[9],
                                                 cam[10], np.array(cam[11]))
        if b == 2:
            yaw -= 0.3 * dx
            pitch = float(np.clip(pitch - 0.3 * dy, -89.0, -2.0))
        else:
            k = dist * 0.0015
            hn = horiz / max(np.linalg.norm(horiz), 1e-9)
            vn = vert / max(np.linalg.norm(vert), 1e-9)
            target = target - hn * dx * k + vn * dy * k
        p.resetDebugVisualizerCamera(dist, yaw, pitch, target.tolist(), physicsClientId=self.cid)

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
                f"{'FAST throw (windup/coast/brake)' if pl.get('kind') == 'fast' else 'gentle throw'}, "
                f"peak joint speed {pl['peak']:.2f} rad/s (limit {self.pl.space.max_vel})",
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
        T, q_at = plan["T"], plan["q_at"]
        pts = []
        for s in np.linspace(0, T, 25):
            self._set_arm(q_at(s))
            pts.append(self._grip())
        for a, b in zip(pts[:-1], pts[1:]):
            self._line(a, b, GREY, 2, self.plan_items)
        if physics_done and plan.get("sim_landing") is not None:
            self._cross((*plan["sim_landing"], tz), RED, 0.05, self.plan_items)
        self.preview = (q_at, T, plan["t_release"], rel["p0"], rel["v0"], plan["residual"],
                        rel.get("t_flight", 1.0))
        self.preview_t0 = time.time()

    # ---- planning (background thread) ---------------------------------

    def click(self, target_xy):
        if self.busy:
            return
        self.busy = True
        self.status = f"Planning for ({target_xy[0]:.2f}, {target_xy[1]:.2f})..."
        print(f"click at ({target_xy[0]:.2f}, {target_xy[1]:.2f}) m (robot frame: x forward, y left)", flush=True)
        self._draw_text()
        threading.Thread(target=self._plan_worker, args=(np.array(target_xy),), daemon=True).start()

    def _plan_worker(self, target):
        try:
            self.pl.residual = np.zeros(2)
            x, land, miss = self.pl.solve(target)
            why = None
            if x is None or miss > 0.05:
                why = "out of reach"
            elif self.pl.min_clearance(x) < 0.03:
                why = "arm too close to the body"
            if why is None:
                quick = self._as_plan(x, target)
                self.pending = ("quick", quick)
                full = self.pl.plan(target)
                if full["ok"]:
                    self.pending = ("full", self._with_motion(full))
                    return
                why = full["reason"]
            if self.fast_fallback:
                self._fast_fallback(target, why)
            else:
                self.pending = ("fail", f"No gentle throw: {why}.")
        except Exception as err:  # keep the window alive
            self.pending = ("fail", f"error: {err}")
        finally:
            self.busy = False

    pending = None

    def _fast_fallback(self, target, why):
        """The gentle throw can't do it: plan a fast throw (twin/fast_throw.py)."""
        from .fast_throw import FastThrowPlanner, FastThrowSpace
        self.status = f"Gentle throw: {why}. Planning a FAST throw (~30 s)..."
        self.pending = ("status", self.status)
        if self.fast is None:
            space = FastThrowSpace(max_vel=self.pl.space.max_vel, max_acc=self.pl.space.max_acc,
                                   gripper_delay=self.pl.gripper_delay)
            self.fast = FastThrowPlanner(space, base=self.pl)
        fp = self.fast
        plan = fp.plan(target)
        if plan is None:
            self.pending = ("fail", f"No throw reaches ({target[0]:.2f}, {target[1]:.2f}) "
                                    f"at {fp.space.max_vel} rad/s (gentle: {why}; fast: none found).")
            return
        T, rf, (times, pos, vel) = fp.space.build(plan["x"])
        pr = fp.predict(plan["x"])
        land = np.asarray(pr["landing"]) if pr["landing"] is not None else np.asarray(plan["sim_landing"])
        self.pending = ("full", {
            "ok": True, "kind": "fast", "fast_plan": plan, "x": np.asarray(plan["x"]), "target": target,
            "T": T, "t_release": rf * T + fp.space.gripper_delay,
            "q_at": lambda t: np.array([np.interp(t, times, pos[:, j]) for j in range(6)]),
            "start_pose": pos[0], "end_pose": pos[-1],
            "release": {**pr, "t_release": rf * T + fp.space.gripper_delay},
            "residual": np.asarray(plan["sim_landing"]) - land,
            "sim_landing": np.asarray(plan["sim_landing"]),
            "params": {"duration": T, "release_fraction": rf}, "peak": plan["peak_joint_speed"]})

    def _with_motion(self, plan):
        """Gentle plans: add the same motion fields the fast plans carry."""
        start, end, T = plan["start_pose"], plan["end_pose"], plan["params"]["duration"]
        return {**plan, "kind": "gentle", "T": T, "t_release": plan["release"]["t_release"],
                "q_at": lambda t: cubic_state(start, end, T, t), "peak": self.pl.peak_speed(plan["x"])}

    def _as_plan(self, x, target):
        start, end, P = self.pl._poses(x)
        return self._with_motion({"ok": True, "x": x, "params": P, "start_pose": start, "end_pose": end,
                                  "release": self.pl.predict(x), "residual": self.pl.residual.copy(),
                                  "target": target, "sim_landing": None})

    def _consume_pending(self):
        if self.pending is None:
            return
        kind, data = self.pending
        self.pending = None
        if kind == "fail":
            self.status = data
            print(f"  -> {data}", flush=True)
        elif kind == "status":
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
        q_at, T, t_rel, p0, v0, residual, t_flight = self.preview
        slow = 0.5
        t = ((time.time() - self.preview_t0) * slow) % (T + 1.5)
        self._set_arm(q_at(min(t, T)))
        if t < t_rel:
            ball = self._grip()
        else:
            s = t - t_rel
            ball = p0 + v0 * s + np.array([0, 0, -4.905 * s * s])
            ball[:2] += residual * min(1.0, s / max(t_flight, 1e-3))
            ball[2] = max(ball[2], self.pl.floor_z + self.pl.radius)
        p.resetBasePositionAndOrientation(self.ball, ball, (0, 0, 0, 1), physicsClientId=self.cid)

    def export(self, plan_dir=None):
        """plan_dir: where to save (default twin_data/plans; tests pass a temp dir
        so 'run_twin.py throw' never picks a test plan as the newest)."""
        if self.plan is None or not self.plan.get("ok"):
            self.status = "Nothing to export yet."
            return None
        plan_dir = PLAN_DIR if plan_dir is None else Path(plan_dir)
        plan_dir.mkdir(parents=True, exist_ok=True)
        if self.plan.get("kind") == "fast":
            from .fast_throw import save_fast_plan
            path = plan_dir / f"fastplan_{time.strftime('%Y%m%d_%H%M%S')}.json"
            save_fast_plan(self.plan["fast_plan"], self.fast.space, path)
        else:
            path = plan_dir / f"plan_{time.strftime('%Y%m%d_%H%M%S')}.json"
            save_plan({**self.plan, "gripper_delay": self.pl.gripper_delay}, path)
        self.status = f"Exported {path.name}"
        print(f"exported {path}\n  on Spot:   python run_twin.py throw --record   (uses the newest plan)")
        return path

    def run(self):
        while p.isConnected(self.cid):
            for ev in p.getMouseEvents(physicsClientId=self.cid):
                etype, mx, my, button, state = ev
                self._camera_mouse(etype, mx, my, button, state)
                if etype == 1 and self.drag is None:
                    self._show_cursor(mx, my)
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
