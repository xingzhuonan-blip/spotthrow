"""Real-time view of a live session (PyBullet window).

While throws run in a background thread, the window shows, ~30 times a second:
  * Spot (body pose from OptiTrack, arm joints from the robot)
  * every unlabeled OptiTrack marker (small grey dots: stray reflections too)
  * the tracked ball (orange) and its trail
After each throw: the predicted flight (blue), measured / predicted landing
(red / blue X), all previous landings (purple) and the target (green).

Drawn in the robot frame of the latest throw (before the first one: Spot's
flat body frame when the window opened). Keys: q = close the window.
"""

import queue
import threading
import time
from collections import deque

import numpy as np
import pybullet as p

from .geometry import flatten, inv_T, transform_points, T_to_pos_quat
from .kinematics import ARM_JOINTS, GRASP_OFFSET, LEGS, ROOT, URDF, WRIST_LINK, place_body, root_to_body
from .mocap import BallTracker, body_pose_world

ORANGE, GREY, BLUE, RED, GREEN, PURPLE = (1, .5, 0), (.55, .55, .55), (.1, .4, 1), (1, .1, .1), (0, .7, 0), (.6, .2, .8)
FPS = 30.0


class LiveView:
    def __init__(self, robot, client, T_rb_body, rb_id, target=None, gui=True):
        self.robot_if, self.client = robot, client
        self.T_rb_body, self.rb_id = np.asarray(T_rb_body, float), rb_id
        self.target = None if target is None else np.array([target[0], target[1]], float)
        self.gui = gui
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.cid)
        self.plane = p.loadURDF(str(ROOT / "plane.urdf"), physicsClientId=self.cid)
        self.robot = p.loadURDF(URDF, physicsClientId=self.cid)
        for i, q in enumerate(LEGS):
            p.resetJointState(self.robot, 6 + i, q, physicsClientId=self.cid)
        self.T_root_body = root_to_body(self.cid, self.robot)
        self.ball = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_SPHERE, radius=0.034, rgbaColor=(*ORANGE, 1),
                                                                 physicsClientId=self.cid), physicsClientId=self.cid)
        p.resetDebugVisualizerCamera(2.4, 50, -20, (0.6, 0, 0.0), physicsClientId=self.cid)

        self.T_world_display = None        # display frame = robot frame of the latest throw
        self.T_world_body = None
        self.tracker = BallTracker()
        self.trail = deque(maxlen=int(3 * FPS))
        self.landings = []                 # (measured landing xy, miss) of all throws so far
        self.results = queue.Queue()
        self.result_items = []
        self.items = {}                    # name -> debug item id, replaced every frame
        self.frames_shown = self.frames_with_ball = self.results_shown = 0

    # ---- called from the worker thread ---------------------------------

    def post_result(self, rec, analysis, entry=None, T_world_robot=None):
        self.results.put((rec, analysis, entry, T_world_robot))

    # ---- drawing helpers (main thread only) ----------------------------

    def _replace(self, name, add):
        old = self.items.get(name, -1)
        new = add(old)
        self.items[name] = new

    def _points(self, name, pts, color, size):
        pts = [list(map(float, q)) for q in pts] or [[0.0, 0.0, -10.0]]
        self._replace(name, lambda old: p.addUserDebugPoints(pts, [color] * len(pts), size,
                                                             replaceItemUniqueId=old, physicsClientId=self.cid))

    def _text(self, name, text, pos, color=(0, 0, 0), size=1.1):
        self._replace(name, lambda old: p.addUserDebugText(text, pos, color, size, replaceItemUniqueId=old,
                                                           physicsClientId=self.cid))

    def _cross(self, pos, color, size=0.06, store=None):
        x, y, z = pos
        ids = [p.addUserDebugLine((x - size, y, z), (x + size, y, z), color, 4, physicsClientId=self.cid),
               p.addUserDebugLine((x, y - size, z), (x, y + size, z), color, 4, physicsClientId=self.cid)]
        if store is not None:
            store.extend(ids)

    # ---- one frame -----------------------------------------------------

    def update(self):
        frame = self.client.latest()
        t_q, q = self.robot_if.joint_state()
        if frame is not None:
            T = body_pose_world(frame, self.rb_id, self.T_rb_body)
            if T is not None:
                self.T_world_body = T
        if self.T_world_body is None:
            self._text("status", "waiting for Spot's body rigid body...", (0, -1.0, 1.2), RED)
            return
        if self.T_world_display is None:
            self.T_world_display = flatten(self.T_world_body)
        T_disp_world = inv_T(self.T_world_display)
        place_body(self.cid, self.robot, self.T_root_body, *T_to_pos_quat(T_disp_world @ self.T_world_body))
        for j, qj in zip(ARM_JOINTS, q):
            p.resetJointState(self.robot, j, qj, physicsClientId=self.cid)
        wpos, worn = p.getLinkState(self.robot, WRIST_LINK, computeForwardKinematics=True,
                                    physicsClientId=self.cid)[4:6]
        grip = np.array(p.multiplyTransforms(wpos, worn, GRASP_OFFSET, (0, 0, 0, 1))[0])

        cands = np.array(frame.unlabeled()).reshape(-1, 3) if frame is not None else np.zeros((0, 3))
        cands = transform_points(T_disp_world, cands) if len(cands) else cands
        ball = self.tracker.step(frame.t_local, cands, grip) if frame is not None else None
        self.frames_shown += 1
        if ball is not None:
            self.frames_with_ball += 1
            self.trail.append(ball)
            p.resetBasePositionAndOrientation(self.ball, ball, (0, 0, 0, 1), physicsClientId=self.cid)
        else:
            p.resetBasePositionAndOrientation(self.ball, (0, 0, -10), (0, 0, 0, 1), physicsClientId=self.cid)
        self._points("markers", cands, GREY, 5)
        self._points("trail", list(self.trail), ORANGE, 4)
        status = "ball tracked" if ball is not None else "ball NOT tracked (put it in the gripper)"
        self._text("status", f"{status} | {len(cands)} unlabeled marker(s)", (0, -1.0, 1.2),
                   GREEN if ball is not None else RED)

    def show_result(self, rec, a, entry, T_world_robot):
        from .viewer import summary_lines

        if T_world_robot is not None:
            self.T_world_display = T_world_robot   # record coordinates == display coordinates
        p.resetBasePositionAndOrientation(self.plane, (0, 0, rec.floor_z), (0, 0, 0, 1), physicsClientId=self.cid)
        for item in self.result_items:
            p.removeUserDebugItem(item, physicsClientId=self.cid)
        self.result_items = []
        if a.get("release_t") is not None:
            pts = a["pred_pos"][::10]
            for p0, p1 in zip(pts[:-1], pts[1:]):
                self.result_items.append(p.addUserDebugLine(p0, p1, BLUE, 2, physicsClientId=self.cid))
            self._cross(a["pred_landing"], BLUE, store=self.result_items)
        if a.get("meas_landing") is not None:
            self._cross(a["meas_landing"], RED, store=self.result_items)
            self.landings.append(np.array(a["meas_landing"]))
            p.addUserDebugPoints([self.landings[-1].tolist()], [PURPLE], 10, physicsClientId=self.cid)
        if self.target is not None:
            self._cross((*self.target, rec.floor_z), GREEN, 0.1, store=self.result_items)
        lines = summary_lines(rec, a)
        if entry is not None:
            lines.append(f"optimizer round {len(self.landings)}: miss {entry['miss'] * 100:.1f} cm")
        for i, text in enumerate(lines):
            self.result_items.append(p.addUserDebugText(text, (0, -1.4, 1.6 - 0.09 * i), (0, 0, 0), 1.0,
                                                        physicsClientId=self.cid))
        self.trail.clear()
        self.results_shown += 1

    # ---- main loop -----------------------------------------------------

    def run(self, worker, max_seconds=None):
        """Run worker(view) in a background thread and draw until it finishes
        (headless) or the window is closed / q is pressed (GUI)."""
        th = threading.Thread(target=self._guard, args=(worker,), daemon=True)
        th.start()
        t_start = time.time()
        try:
            while p.isConnected(self.cid):
                if self.gui:
                    keys = p.getKeyboardEvents(physicsClientId=self.cid)
                    if any(k == ord("q") and s & p.KEY_WAS_TRIGGERED for k, s in keys.items()):
                        break
                self.update()
                while not self.results.empty():
                    self.show_result(*self.results.get())
                if not th.is_alive() and (not self.gui or self.worker_error):
                    break
                if max_seconds is not None and time.time() - t_start > max_seconds:
                    break
                time.sleep(1.0 / FPS)
        finally:
            if p.isConnected(self.cid):
                p.disconnect(self.cid)
        th.join(timeout=0.1)
        if self.worker_error:
            raise self.worker_error

    worker_error = None

    def _guard(self, worker):
        try:
            worker(self)
        except BaseException as err:  # surface worker failures in the main thread
            self.worker_error = err
