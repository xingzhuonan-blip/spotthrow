"""Calibration of the OptiTrack rigid body on Spot's body.

The markers stuck on Spot define a rigid body whose frame (pivot) is wherever
Motive put it, not Spot's body frame. We need the fixed transform T_rb_body
(Spot body frame expressed in the rigid-body frame):

    T_world_body = T_world_rb (streamed by Motive) @ T_rb_body

How: hold the ball (a single marker) in the gripper and move the arm through
a few poses. For each pose we know
  * the ball in the BODY frame, from forward kinematics of the arm joints;
  * the ball in the WORLD frame, measured by OptiTrack;
  * T_world_rb, measured by OptiTrack.
Moving the ball into the rigid-body frame gives point pairs related by
T_rb_body only, solved with Kabsch. Needs >= 3 non-collinear poses; use 6-10.
"""

import json
from pathlib import Path

import numpy as np

from .geometry import inv_T, kabsch, transform_points

# Arm poses (SDK joint order) for calibration. The gripper points straight
# up (sh1 + el0 + wr0 = -1.62) so the ball sticks out of the jaws where the
# cameras can see it (with the gripper pointing forward the ball was hidden).
# Gripper 0.5-0.8 m in front, +-0.2 m to the sides, 0.7-1.1 m above the body.
# Checked in simulation for joint limits and collisions (see
# tests/test_twin_live.py); still run them SLOWLY first.
CALIBRATION_POSES = [
    [0.0, -1.35, 0.35, 0.0, -0.62, 0.0],
    [0.5, -1.35, 0.35, 0.0, -0.62, 0.0],
    [-0.5, -1.35, 0.35, 0.0, -0.62, 0.0],
    [0.0, -1.2, 0.9, 0.0, -1.32, 0.0],
    [0.4, -1.2, 0.9, 0.0, -1.32, 0.0],
    [-0.4, -1.2, 0.9, 0.0, -1.32, 0.0],
    [0.0, -1.9, 2.0, 0.0, -1.72, 0.0],   # lower and nearer (breaks the reach/height line)
    [0.3, -1.0, 0.3, 0.0, -0.92, 0.0],   # higher and farther
]


def solve_rb_to_body(ball_body, ball_world, T_world_rb):
    """ball_body (N,3): FK gripper centre in the body frame.
    ball_world (N,3): OptiTrack ball position. T_world_rb: list of N 4x4.
    Returns (T_rb_body, rms residual in metres)."""
    ball_rb = np.array([transform_points(inv_T(T), p[None])[0] for T, p in zip(T_world_rb, ball_world)])
    T = kabsch(ball_body, ball_rb)
    rms = float(np.sqrt(np.mean(np.sum((transform_points(T, ball_body) - ball_rb) ** 2, axis=1))))
    return T, rms


def save(path, T_rb_body, rb_id, up_axis, rms):
    Path(path).write_text(json.dumps({"body_rigid_body_id": rb_id, "up_axis": up_axis,
                                      "T_rb_body": np.asarray(T_rb_body).tolist(), "rms_m": rms}, indent=2))


def load(path):
    d = json.loads(Path(path).read_text())
    return np.array(d["T_rb_body"]), d["body_rigid_body_id"], d["up_axis"], d["rms_m"]


def fixed_body_from_heading_point(T_world_rb, point_world):
    """Spot does not move: estimate its body frame without the arm.

    Origin = the body rigid body's position (marker centre, so a few cm off
    Spot's true body centre). Heading = from there towards a point placed on
    Spot's forward centre line (the ball, about 1 m in front). Level (z up).
    Returns T_rb_body, so the rest of the pipeline works unchanged."""
    from .geometry import rot_z
    origin = np.asarray(T_world_rb, float)[:3, 3]
    d = np.asarray(point_world, float)[:2] - origin[:2]
    if np.linalg.norm(d) < 0.3:
        raise ValueError("Heading point is too close to the body (need it about 1 m in front).")
    T_world_body = np.eye(4)
    T_world_body[:3, :3] = rot_z(np.arctan2(d[1], d[0]))
    T_world_body[:3, 3] = origin
    return inv_T(T_world_rb) @ T_world_body
