"""PyBullet viewer: replay recorded throws with the analysis overlaid.

  orange dots   measured ball positions (OptiTrack)
  grey line     gripper path (from the arm joints)
  yellow        estimated release point
  blue line     flight predicted from the release state alone
  red / blue X  measured / predicted landing point
  green X       target (if the record has one)

Keys (click the 3D view first):
  space  pause / play        left / right  step back / forward (when paused)
  r      restart             n / p         next / previous throw
  q      quit
"""

import time

import numpy as np
import pybullet as p

from . import ballistics
from .kinematics import ARM_JOINTS, LEGS, URDF, ROOT, place_body, root_to_body
from .record import ThrowRecord

FPS = 60.0
ORANGE, GREY, YELLOW, BLUE, RED, GREEN = (1, .5, 0), (.5, .5, .5), (1, .9, 0), (.1, .4, 1), (1, .1, .1), (0, .8, 0)


def summary_lines(rec, a):
    P = rec.params
    names = ("sh0", "sh1", "el0", "el1", "wr0", "wr1")
    moving = ", ".join(names[j] for j in P.get("moving_joints", [])) or "?"
    lines = [f"{rec.source}: T={P.get('duration', float('nan')):.2f} s, release at "
             f"{P.get('release_fraction', float('nan')):.2f} of swing, "
             f"ball={P.get('ball')}, moving joints: {moving}"]
    if a.get("release_t") is None:
        return lines + ["release not detected"]
    lines.append(f"open cmd {rec.release_cmd_t:.3f} s -> ball free {a['release_t']:.3f} s "
                 f"(delay {1000 * (a['release_t'] - rec.release_cmd_t):.0f} ms)")
    lines.append(f"release speed {a['speed']:.2f} m/s, elevation {a['elevation_deg']:.0f} deg, "
                 f"heading {a['heading_deg']:.0f} deg")
    if a.get("meas_landing") is not None:
        lines.append(f"landing: measured {np.round(a['meas_landing'][:2], 2)}, predicted "
                     f"{np.round(a['pred_landing'][:2], 2)}, error {100 * a['landing_error']:.1f} cm")
    if not np.isnan(rec.target).any() and a.get("meas_landing") is not None:
        lines.append(f"target {np.round(rec.target[:2], 2)}: miss "
                     f"{100 * np.linalg.norm(a['meas_landing'][:2] - rec.target[:2]):.1f} cm")
    cd = f"{a['fitted_cd']:.2f}" if a.get("cd_identifiable") else "not identifiable (flight too short/slow)"
    lines.append(f"fitted drag coeff: {cd}")
    return lines


class Viewer:
    def __init__(self, files, gui=True):
        self.files = list(files)
        self.gui = gui
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.cid)
        # Everything is drawn in the robot frame (see record.py); the floor is
        # moved to floor_z per record.
        self.plane = p.loadURDF(str(ROOT / "plane.urdf"), physicsClientId=self.cid)
        self.robot = p.loadURDF(URDF, physicsClientId=self.cid)
        for i, q in enumerate(LEGS):
            p.resetJointState(self.robot, 6 + i, q, physicsClientId=self.cid)
        self.T_root_body = root_to_body(self.cid, self.robot)
        self.ball = None
        self.speed_slider = p.addUserDebugParameter("playback speed", 0.05, 2.0, 0.25,
                                                    physicsClientId=self.cid)
        p.resetDebugVisualizerCamera(2.2, 60, -15, (0.5, 0, 0.2), physicsClientId=self.cid)
        self.idx = 0
        self.load(0)

    # ---- per-record setup -------------------------------------------------

    def load(self, idx):
        self.idx = idx % len(self.files)
        self.rec = ThrowRecord.load(self.files[self.idx])
        P = self.rec.params
        self.a = ballistics.analyze(self.rec, P["ball_radius"], P["ball_mass"], P["drag_coeff"])
        p.removeAllUserDebugItems(physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.plane, (0, 0, self.rec.floor_z), (0, 0, 0, 1),
                                          physicsClientId=self.cid)
        if self.ball is not None:
            p.removeBody(self.ball, physicsClientId=self.cid)
        r = P["ball_radius"]
        self.ball = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=(*ORANGE, 1),
                                                                 physicsClientId=self.cid),
                                      physicsClientId=self.cid)
        self.draw_static()
        self.t = 0.0
        self.t_end = self.rec.ball_t[-1]
        self.paused = False
        print(f"\n[{self.idx + 1}/{len(self.files)}] {self.files[self.idx]}")
        for line in summary_lines(self.rec, self.a):
            print("  " + line)

    def line(self, a, b, color, width=2):
        p.addUserDebugLine(a, b, color, width, physicsClientId=self.cid)

    def cross(self, pos, color, size=0.06):
        x, y, z = pos
        self.line((x - size, y, z), (x + size, y, z), color, 4)
        self.line((x, y - size, z), (x, y + size, z), color, 4)

    def polyline(self, pts, color, width=2):
        for a, b in zip(pts[:-1], pts[1:]):
            self.line(a, b, color, width)

    def draw_static(self):
        rec, a = self.rec, self.a
        ok = ~np.isnan(rec.ball_pos).any(axis=1)
        p.addUserDebugPoints(rec.ball_pos[ok].tolist(), [ORANGE] * int(ok.sum()), 4, physicsClientId=self.cid)
        self.polyline(a["grip"][::4], GREY, 1)
        if a.get("release_t") is not None:
            p.addUserDebugPoints([a["p0"].tolist()], [YELLOW], 12, physicsClientId=self.cid)
            self.polyline(a["pred_pos"][::10], BLUE, 2)
            self.cross(a["pred_landing"], BLUE)
            if a.get("meas_landing") is not None:
                self.cross(a["meas_landing"], RED)
        if not np.isnan(rec.target).any():
            self.cross(rec.target, GREEN, 0.1)
        for i, text in enumerate(summary_lines(rec, a)):
            p.addUserDebugText(text, (0.0, -1.2, 1.6 - 0.09 * i), (0, 0, 0), 1.0, physicsClientId=self.cid)

    # ---- playback -----------------------------------------------------------

    def show(self, t):
        place_body(self.cid, self.robot, self.T_root_body, *self.rec.body_at(t))
        q = self.rec.q_at(t)
        for j, qj in zip(ARM_JOINTS, q):
            p.resetJointState(self.robot, j, qj, physicsClientId=self.cid)
        bt, bx = ballistics.valid(self.rec.ball_t, self.rec.ball_pos)
        pos = [np.interp(t, bt, bx[:, k]) for k in range(3)]
        p.resetBasePositionAndOrientation(self.ball, pos, (0, 0, 0, 1), physicsClientId=self.cid)

    def handle_keys(self):
        keys = p.getKeyboardEvents(physicsClientId=self.cid)
        down = {k for k, s in keys.items() if s & p.KEY_WAS_TRIGGERED}
        if ord(" ") in down:
            self.paused = not self.paused
        if ord("r") in down:
            self.t = 0.0
        if ord("n") in down:
            self.load(self.idx + 1)
        if ord("p") in down:
            self.load(self.idx - 1)
        step = 1.0 / 120.0
        if p.B3G_RIGHT_ARROW in down:
            self.t = min(self.t + step, self.t_end)
        if p.B3G_LEFT_ARROW in down:
            self.t = max(self.t - step, 0.0)
        return ord("q") not in down

    def run(self, max_frames=None):
        frame = 0
        while p.isConnected(self.cid):
            if self.gui and not self.handle_keys():
                break
            if not self.paused:
                speed = p.readUserDebugParameter(self.speed_slider, physicsClientId=self.cid) if self.gui else 1.0
                self.t += speed / FPS
                if self.t > self.t_end + 1.0:  # hold the end for a moment, then loop
                    self.t = 0.0
            self.show(min(self.t, self.t_end))
            frame += 1
            if max_frames is not None and frame >= max_frames:
                break
            if self.gui:
                time.sleep(1.0 / FPS)
        if p.isConnected(self.cid):
            p.disconnect(self.cid)
