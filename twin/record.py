"""One throw = one ThrowRecord, saved as a single .npz file.

The same format is used whatever the data source (simulation, or Spot +
OptiTrack), so the viewer and the analysis never need to know where the data
came from.

Coordinate frame of everything in a record: the ROBOT frame, i.e. Spot's
gravity-aligned body frame ("flat body" in the SDK) frozen at the start of
the swing: origin at the body centre, x forward, y left, z up (gravity is -z),
metres. Because it is frozen, the ball flight is plain ballistics in it even
if the body sways during the throw; that sway is kept in body_pos/body_orn.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 2


@dataclass
class ThrowRecord:
    # --- what was commanded ---
    params: dict                       # duration, release_fraction, ball, start/end pose, ... (free-form)
    # --- robot (sim, or robot_state on Spot) ---
    robot_t: np.ndarray                # (N,) s, 0 = swing start
    q_cmd: np.ndarray                  # (N, 6) commanded arm joints, SDK order sh0..wr1
    q: np.ndarray                      # (N, 6) measured arm joints
    body_pos: np.ndarray               # (N, 3) body origin in the robot frame (~0 unless the body moves)
    body_orn: np.ndarray               # (N, 4) body orientation in the robot frame, xyzw
    # --- ball (sim + fake OptiTrack, or real OptiTrack) ---
    ball_t: np.ndarray                 # (M,) s, same clock as robot_t
    ball_pos: np.ndarray               # (M, 3) measured ball positions; NaN rows = marker lost
    # --- scene ---
    floor_z: float                     # floor height in the robot frame (negative)
    release_cmd_t: float               # when the gripper open command was sent
    source: str = "sim"                # "sim" | "live"
    target: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))  # optional target point
    # Ground truth, only known in simulation (NaN for real data).
    true_release_t: float = float("nan")
    true_landing: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, format_version=FORMAT_VERSION, params=json.dumps(self.params), robot_t=self.robot_t,
            q_cmd=self.q_cmd, q=self.q, body_pos=self.body_pos, body_orn=self.body_orn,
            ball_t=self.ball_t, ball_pos=self.ball_pos, floor_z=self.floor_z,
            release_cmd_t=self.release_cmd_t, source=self.source, target=self.target,
            true_release_t=self.true_release_t, true_landing=self.true_landing)
        return path

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        if "format_version" not in d or int(d["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"{path}: old record format, regenerate it (python run_twin.py simulate).")
        return cls(params=json.loads(str(d["params"])), robot_t=d["robot_t"], q_cmd=d["q_cmd"], q=d["q"],
                   body_pos=d["body_pos"], body_orn=d["body_orn"], ball_t=d["ball_t"],
                   ball_pos=d["ball_pos"], floor_z=float(d["floor_z"]),
                   release_cmd_t=float(d["release_cmd_t"]), source=str(d["source"]), target=d["target"],
                   true_release_t=float(d["true_release_t"]), true_landing=d["true_landing"])

    def q_at(self, t):
        """Measured arm joints at time t (linear interpolation)."""
        return np.array([np.interp(t, self.robot_t, self.q[:, j]) for j in range(6)])

    def body_at(self, t):
        """Body pose (pos, quat xyzw) at time t (nearest sample)."""
        i = int(np.clip(np.searchsorted(self.robot_t, t), 0, len(self.robot_t) - 1))
        return self.body_pos[i], self.body_orn[i]
