"""From raw OptiTrack frames to robot-frame data: body pose and ball tracking."""

import numpy as np

from .geometry import make_T, rot_z


def body_pose_world(frame, rb_id, T_rb_body):
    """Spot body pose in the OptiTrack world for one frame, or None if not tracked."""
    rb = frame.rigid_bodies.get(rb_id)
    if rb is None or not rb.valid:
        return None
    return make_T(rb.pos, rb.quat) @ T_rb_body


def robot_frame_from_body_poses(T_list):
    """Robot frame = gravity-aligned body frame, averaged over a few frames to
    beat marker noise: mean position, circular mean of yaw."""
    pos = np.mean([T[:3, 3] for T in T_list], axis=0)
    yaws = [np.arctan2(T[1, 0], T[0, 0]) for T in T_list]
    yaw = np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))
    F = np.eye(4)
    F[:3, :3] = rot_z(yaw)
    F[:3, 3] = pos
    return F


class BallTracker:
    """Picks the ball among the unlabeled markers of each frame.

    Starts on the candidate closest to the gripper (within init_radius), then
    follows the candidate closest to a constant-velocity prediction (within a
    gate). Stray reflections elsewhere in the room are therefore ignored.
    """

    def __init__(self, init_radius=0.10, gate=0.08, max_miss=30, max_gate=0.30):
        self.init_radius = init_radius
        self.gate = gate
        self.max_gate = max_gate  # the gate grows while the ball is lost; cap it so it can't jump to Spot's back
        self.max_miss = max_miss
        self.reset()

    def reset(self):
        self.last_p = self.last_t = None
        self.vel = np.zeros(3)
        self.misses = 0

    def step(self, t, cands, gripper_point):
        """One frame. Returns the ball position, or None if not found.
        Usable online (live view) or offline (track)."""
        cands = np.asarray(cands, float).reshape(-1, 3)
        found = None
        if len(cands) and self.last_p is None:
            g = np.asarray(gripper_point, float)
            if not np.isnan(g).any():
                d = np.linalg.norm(cands - g, axis=1)
                k = int(np.argmin(d))
                if d[k] < self.init_radius:
                    found = self.last_p = cands[k]
                    self.last_t, self.vel, self.misses = t, np.zeros(3), 0
            return found
        if len(cands):
            pred = self.last_p + self.vel * (t - self.last_t)
            # Also accept the ball near where it was last seen: it may have
            # stopped or bounced (the velocity prediction is then wrong).
            d = np.minimum(np.linalg.norm(cands - pred, axis=1), np.linalg.norm(cands - self.last_p, axis=1))
            k = int(np.argmin(d))
            if d[k] < min(self.gate * (1 + self.misses), self.max_gate):
                p = cands[k]
                if t - self.last_t > 0.002:  # tiny steps turn marker noise into huge velocities
                    new_vel = (p - self.last_p) / (t - self.last_t)
                    self.vel = new_vel if self.misses else 0.5 * self.vel + 0.5 * new_vel
                found = self.last_p = p
                self.last_t, self.misses = t, 0
                return found
        if self.last_p is not None:
            self.misses += 1
            if self.misses > self.max_miss:
                self.reset()  # lost: re-acquire only near the gripper
        return None

    def track(self, times, candidates, gripper_points):
        """times (n,), candidates: list of (k_i, 3) arrays, gripper_points: (n, 3)
        (NaN where unknown). Returns (n, 3) ball positions, NaN when not found."""
        self.reset()
        candidates = drop_static(candidates)
        out = np.full((len(times), 3), np.nan)
        for i, (t, cands) in enumerate(zip(times, candidates)):
            p = self.step(t, cands, gripper_points[i])
            if p is not None:
                out[i] = p
        return out


def drop_static(candidates, radius=0.02, frac=0.8):
    """Remove markers that sit still for the whole recording (e.g. reflective
    spots on the box mounted on Spot): the ball moves, so it is never one of
    them. A point counts as static if it is in the first and the last frame and
    within `radius` of that place in at least `frac` of all frames."""
    cs = [np.asarray(c, float).reshape(-1, 3) for c in candidates]
    nonempty = [c for c in cs if len(c)]
    if len(nonempty) < 10:
        return cs
    static = []
    for p in nonempty[0]:
        if not len(nonempty[-1]) or np.linalg.norm(nonempty[-1] - p, axis=1).min() > radius:
            continue
        hits = sum(1 for c in nonempty if np.linalg.norm(c - p, axis=1).min() < radius)
        if hits >= frac * len(nonempty):
            static.append(p)
    if not static:
        return cs
    S = np.array(static)
    return [c[np.linalg.norm(c[:, None] - S[None], axis=2).min(axis=1) > radius] if len(c) else c for c in cs]
