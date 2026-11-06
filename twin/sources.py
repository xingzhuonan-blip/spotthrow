"""Data sources that produce ThrowRecords.

SimSource   - runs the PyBullet replay of a throw trajectory and samples the
              ball like OptiTrack would (rate, noise, dropouts).
LiveSource  - Spot + OptiTrack, see twin/live.py. Same output format.
"""

import numpy as np

from .geometry import flatten, inv_T, make_T, T_to_pos_quat, transform_points
from .record import ThrowRecord


class FakeOptiTrack:
    """Turns the true ball trajectory into what OptiTrack would report."""

    def __init__(self, rate_hz=120.0, noise_m=0.001, dropout=0.02, seed=None):
        self.rate_hz = rate_hz
        self.noise_m = noise_m
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)

    def measure(self, t_true, pos_true):
        t = np.arange(t_true[0], t_true[-1], 1.0 / self.rate_hz)
        pos = np.stack([np.interp(t, t_true, pos_true[:, a]) for a in range(3)], axis=1)
        pos += self.rng.normal(0.0, self.noise_m, pos.shape)
        pos[self.rng.random(len(t)) < self.dropout] = np.nan  # marker lost
        return t, pos


def trajectory_params(times, positions):
    positions = np.asarray(positions)
    moving = [j for j in range(6) if np.ptp(positions[:, j]) > 1e-6]
    return {"start_pose": np.round(positions[0], 4).tolist(), "end_pose": np.round(positions[-1], 4).tolist(),
            "moving_joints": moving}


def world_log_to_robot_frame(log):
    """Sim log (PyBullet world) -> robot frame frozen at swing start.
    Returns T_world_robot and the converted arrays."""
    T_world_robot = flatten(make_T(log["body_pos"][0], log["body_orn"][0]))
    T_robot_world = inv_T(T_world_robot)
    body_pos, body_orn = [], []
    for pos, orn in zip(log["body_pos"], log["body_orn"]):
        bp, bo = T_to_pos_quat(T_robot_world @ make_T(pos, orn))
        body_pos.append(bp)
        body_orn.append(bo)
    floor_z = -T_world_robot[2, 3]  # world floor is z = 0
    return T_world_robot, np.array(body_pos), np.array(body_orn), floor_z


class SimSource:
    def __init__(self, ball="tennis", gripper_delay=0.1, max_vel=10.0, max_acc=100.0, fly_time=2.0,
                 optitrack=None):
        self.ball = ball
        self.gripper_delay = gripper_delay
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.fly_time = fly_time
        self.optitrack = optitrack or FakeOptiTrack()

    def simulate(self, duration, release_fraction, trajectory=None):
        """Raw simulation result (PyBullet world frame)."""
        # Imported here: pulls in PyBullet and spot_throw_safe only when simulating.
        import sim_check_trajectory as sim

        return sim.run(duration, release_fraction, self.gripper_delay, self.ball, self.max_vel,
                       self.max_acc, gui=False, fly_time=self.fly_time, check_collisions=False,
                       verbose=False, trajectory=trajectory)

    def throw(self, duration, release_fraction, target=None, trajectory=None):
        """target: (x, y, z) in the robot frame (z is ignored, the floor is used)."""
        res = self.simulate(duration, release_fraction, trajectory)
        log = res["log"]
        t = np.array(log["t"])
        T_world_robot, body_pos, body_orn, floor_z = world_log_to_robot_frame(log)
        T_robot_world = inv_T(T_world_robot)
        ball_robot = transform_points(T_robot_world, np.array(log["ball"]))
        ball_t, ball_pos = self.optitrack.measure(t, ball_robot)
        rel = res["release_state"]
        landing = res["landing"]
        if trajectory is None:
            import spot_throw_safe as real
            traj_times, traj_pos, _ = real.sample_trajectory(duration)
        else:
            traj_times, traj_pos = trajectory[0], trajectory[1]
        params = {"duration": duration, "release_fraction": release_fraction, "ball": self.ball,
                  "gripper_delay": self.gripper_delay, "ball_radius": res["ball_radius"],
                  "ball_mass": res["ball_mass"], "drag_coeff": res["drag_coeff"],
                  **trajectory_params(traj_times, traj_pos)}
        if target is not None:
            target = np.array([target[0], target[1], floor_z], float)
        return ThrowRecord(
            params=params, robot_t=t, q_cmd=np.array(log["q_cmd"]), q=np.array(log["q"]),
            body_pos=body_pos, body_orn=body_orn, ball_t=ball_t, ball_pos=ball_pos, floor_z=floor_z,
            release_cmd_t=res["release_cmd_t"], source="sim",
            target=np.full(3, np.nan) if target is None else target,
            true_release_t=float("nan") if rel is None else rel[0],
            true_landing=(np.full(3, np.nan) if landing is None
                          else transform_points(T_robot_world, landing[1][None])[0]),
        )
