"""Live data source: Spot (arm joints) + OptiTrack (body rigid body + ball).

Pieces (each replaceable by a fake, see twin/fake_world.py):
  robot   - object with joint_state(), move_arm(q, seconds) and
            run_throw(duration, release_fraction, trajectory)
            -> {"start_time", "release_cmd_time", "joint_t", "q"} (local clock)
  client  - twin.natnet.NatNetClient, already started
  calib   - T_rb_body from twin.calibration (body markers -> Spot body frame)

Timing: OptiTrack frames are stamped with the local arrival time, joint states
with the local time they were read. Both are a few ms late; fine for now,
measure the latency later if needed.
"""

import threading
import time

import numpy as np

from throw_trajectory import ARM_JOINT_NAMES

from . import calibration
from .geometry import inv_T, make_T, T_to_pos_quat, transform_points
from .kinematics import ArmFK
from .mocap import BallTracker, body_pose_world, robot_frame_from_body_poses
from .record import ThrowRecord
from .sources import trajectory_params


class SpotRobotInterface:
    """Real robot, wrapping spot_throw_safe.SpotThrows (already powered on and standing)."""

    def __init__(self, ctl, max_vel=2.5, max_acc=10.0):
        self.ctl = ctl
        self.max_vel = max_vel
        self.max_acc = max_acc

    def joint_state(self):
        state = self.ctl._state_client.get_robot_state()
        t = time.time()
        jmap = {js.name: js.position.value for js in state.kinematic_state.joint_states}
        return t, np.array([jmap[n] for n in ARM_JOINT_NAMES])

    def move_arm(self, q, seconds=4.0):
        from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
        cmd = RobotCommandBuilder.arm_joint_move_helper(joint_positions=[list(q)], times=[seconds],
                                                        max_vel=1.0, max_acc=2.0)
        cmd_id = self.ctl._command_client.robot_command(cmd)
        if not block_until_arm_arrives(self.ctl._command_client, cmd_id, timeout_sec=seconds + 6.0):
            raise RuntimeError("Arm did not reach the requested pose.")

    def run_throw(self, duration, release_fraction, trajectory=None):
        log_t, log_q = [], []
        stop = threading.Event()

        def poll():  # as fast as robot_state RPCs allow (typically 50-100 Hz)
            while not stop.is_set():
                t, q = self.joint_state()
                log_t.append(t)
                log_q.append(q)

        th = threading.Thread(target=poll, daemon=True)
        th.start()
        try:
            time.sleep(0.3)
            info = self.ctl.throw(T=duration, max_vel=self.max_vel, max_acc=self.max_acc, release=True,
                                  release_fraction=release_fraction, trajectory=trajectory)
            time.sleep(0.5)
        finally:
            stop.set()
            th.join()
        return {**info, "joint_t": np.array(log_t), "q": np.array(log_q)}


def mocap_times(frames):
    """Local-clock time of each frame, from Motive's timestamps (see build_record)."""
    t_local = np.array([f.t_local for f in frames])
    ts = np.array([f.timestamp for f in frames])
    if len(ts) > 1 and np.all(np.diff(ts) > 0):
        return ts + np.min(t_local - ts)
    return t_local  # no usable server timestamps: fall back to arrival times


def _interp_q(joint_t, q, t):
    return np.array([np.interp(t, joint_t, q[:, j]) for j in range(6)])


class LiveSource:
    def __init__(self, robot, client, T_rb_body, body_rb_id, ball="tennis", fly_time=2.0, tracker=None,
                 floor_z_world=0.0):
        from sim_spot_throwing import Spot  # ball parameters only

        self.robot = robot
        self.client = client
        self.T_rb_body = np.asarray(T_rb_body, float)
        self.rb_id = body_rb_id
        self.ball = ball
        self.radius, self.mass, self.drag_coeff = Spot.ball_parameters[ball]
        self.fly_time = fly_time
        self.tracker = tracker or BallTracker()
        self.floor_z_world = floor_z_world  # Motive ground plane = floor
        self.last_T_world_robot = None

    def throw(self, duration, release_fraction, target=None, trajectory=None):
        if trajectory is None:
            import spot_throw_safe as real
            trajectory = real.sample_trajectory(duration)
        times, positions, velocities = (np.asarray(x, float) for x in trajectory)
        info = self.robot.run_throw(duration, release_fraction, trajectory)
        time.sleep(self.fly_time)
        frames = self.client.frames_between(info["start_time"] - 1.0, time.time())
        return self.build_record(frames, info, times, positions, duration, release_fraction, target)

    def build_record(self, frames, info, traj_t, traj_pos, duration, release_fraction, target=None):
        t0 = info["start_time"]
        # Only frames while the joints were being logged (earlier ones may show
        # the arm in a previous pose).
        frames = [f for f in frames if f.t_local >= info["joint_t"][0]]
        if not frames:
            raise RuntimeError("No OptiTrack frames received during the throw.")

        # --- frame times: Motive's own clock, shifted onto the local clock ---
        # Arrival times are bursty (network, busy threads); Motive's timestamps
        # are regular. Arrival is never earlier than capture, so the smallest
        # (arrival - timestamp) is the best clock offset.
        frame_t = mocap_times(frames)

        # --- robot frame: body pose just before the swing ---
        poses = [(t, body_pose_world(f, self.rb_id, self.T_rb_body)) for t, f in zip(frame_t, frames)]
        before = [T for t, T in poses if T is not None and t0 - 0.3 <= t <= t0]
        if not before:
            # Arm hides the body markers. Spot does not walk between throws,
            # so use the latest tracked body pose from the buffer.
            with self.client._lock:
                buffered = list(self.client.frames)
            tracked = [body_pose_world(f, self.rb_id, self.T_rb_body) for f in buffered]
            tracked = [T for T in tracked if T is not None][-36:]
            if not tracked:
                raise RuntimeError(f"Rigid body {self.rb_id} (Spot body) never tracked; check Motive.")
            print("  body hidden before the swing; using its last tracked pose")
            before = tracked
        T_world_robot = robot_frame_from_body_poses(before)
        self.last_T_world_robot = T_world_robot  # for the live view
        T_robot_world = inv_T(T_world_robot)

        # --- arm joints (measured), commanded joints from the trajectory ---
        joint_t, q = np.asarray(info["joint_t"]), np.asarray(info["q"])
        robot_t = joint_t - t0
        q_cmd = np.array([[np.interp(t, traj_t, traj_pos[:, j]) for j in range(6)] for t in robot_t])

        # --- body pose per joint sample, in the robot frame ---
        last = T_world_robot
        body_world = []
        for T in (T for _, T in poses):
            last = T if T is not None else last
            body_world.append(last)
        body_pos, body_orn = [], []
        for t in joint_t:
            i = int(np.clip(np.searchsorted(frame_t, t), 0, len(frame_t) - 1))
            bp, bo = T_to_pos_quat(T_robot_world @ body_world[i])
            body_pos.append(bp)
            body_orn.append(bo)

        # --- ball: track among unlabeled markers, seeded at the gripper ---
        fk = ArmFK()
        try:
            grip = np.array([fk.grasp_point(_interp_q(joint_t, q, t), *T_to_pos_quat(T_b))
                             for t, T_b in zip(frame_t, body_world)])
        finally:
            fk.close()
        cands = [np.array(f.unlabeled()) for f in frames]
        ball_world = self.tracker.track(frame_t, cands, grip)
        if getattr(self, "raw_path", None):
            # Raw points for diagnosis: all unlabeled markers per frame and
            # the gripper position the tracker started from (world frame).
            np.savez(self.raw_path, t=frame_t - t0, grip_world=grip, ball_world=ball_world,
                     cands=np.array([c.reshape(-1, 3) for c in cands] + [None], dtype=object)[:-1],
                     T_world_robot=T_world_robot)
        ok = ~np.isnan(ball_world).any(axis=1)
        ball_robot = np.full_like(ball_world, np.nan)
        ball_robot[ok] = transform_points(T_robot_world, ball_world[ok])

        floor_z = self.floor_z_world - T_world_robot[2, 3]
        params = {"duration": duration, "release_fraction": release_fraction, "ball": self.ball,
                  "ball_radius": self.radius, "ball_mass": self.mass, "drag_coeff": self.drag_coeff,
                  "body_rigid_body_id": self.rb_id, **trajectory_params(traj_t, traj_pos)}
        rel = info.get("release_cmd_time")
        return ThrowRecord(
            params=params, robot_t=robot_t, q_cmd=q_cmd, q=q, body_pos=np.array(body_pos),
            body_orn=np.array(body_orn), ball_t=frame_t - t0, ball_pos=ball_robot, floor_z=floor_z,
            release_cmd_t=float("nan") if rel is None else rel - t0, source="live",
            target=(np.full(3, np.nan) if target is None
                    else np.array([target[0], target[1], floor_z], float)))


def _rb_pose(frames, rb_id):
    rbs = [f.rigid_bodies[rb_id] for f in frames if rb_id in f.rigid_bodies and f.rigid_bodies[rb_id].valid]
    if not rbs:
        return None
    return make_T(np.mean([r.pos for r in rbs], axis=0), rbs[len(rbs) // 2].quat)


def run_calibration(robot, client, rb_id, poses=None, settle_s=1.0, average_s=0.5, verbose=True,
                    min_poses=5, save_path=None):
    """Move the arm (holding the ball) through calibration poses and solve
    T_rb_body. Returns (T_rb_body, rms, per-pose data).

    Spot must not walk during calibration. When the arm hides the body
    markers, the body pose measured before the arm moved is used instead.
    Poses where the ball is not seen are skipped. With save_path, the raw
    data is saved after every pose."""
    poses = calibration.CALIBRATION_POSES if poses is None else poses
    t_start = time.time()
    time.sleep(average_s)
    T_ref = _rb_pose(client.frames_between(t_start, time.time()), rb_id)
    if T_ref is None:
        raise RuntimeError(f"Rigid body {rb_id} not tracked before the arm moves.")
    fk = ArmFK()
    ball_body, candidates, T_world_rb, used, skipped = [], [], [], [], []

    def save():
        if save_path is not None:
            np.savez(save_path, ball_body=np.array(ball_body), T_world_rb=np.array(T_world_rb),
                     T_ref=T_ref, used_poses=np.array(used), skipped_poses=np.array(skipped),
                     candidates=np.array([np.array(c) for c in candidates] + [None], dtype=object)[:-1])

    try:
        for k, pose in enumerate(poses):
            robot.move_arm(pose)
            time.sleep(settle_s)
            t_start = time.time()
            time.sleep(average_s)
            frames = client.frames_between(t_start, time.time())
            _, q = robot.joint_state()
            T_wrb = _rb_pose(frames, rb_id)
            body_note = "body tracked"
            if T_wrb is None:
                T_wrb, body_note = T_ref, "body hidden, using pose from before"
            clusters = _stable_points([f.unlabeled() for f in frames])
            if not clusters:
                skipped.append(k)
                if verbose:
                    print(f"  pose {k + 1}/{len(poses)}: no unlabeled marker (ball) visible, skipped")
                save()
                continue
            candidates.append(clusters)
            ball_body.append(fk.grasp_point(q))
            T_world_rb.append(T_wrb)
            used.append(k)
            if verbose:
                print(f"  pose {k + 1}/{len(poses)}: {len(clusters)} unlabeled marker(s), {body_note}, "
                      f"{len(frames)} frames")
            save()
    finally:
        fk.close()
    if len(used) < min_poses:
        raise RuntimeError(f"Only {len(used)} usable poses (need {min_poses}); skipped {skipped}.")
    ball_world = _pick_ball(candidates, T_world_rb)
    seen = ~np.isnan(ball_world).any(axis=1)
    for k in np.array(used)[~seen]:
        if verbose:
            print(f"  pose {k + 1}: only static points (not the ball), dropped")
    used = list(np.array(used)[seen])
    ball_body = list(np.array(ball_body)[seen])
    T_world_rb = list(np.array(T_world_rb)[seen])
    ball_world = ball_world[seen]
    candidates = [c for c, ok in zip(candidates, seen) if ok]
    if len(used) < min_poses:
        raise RuntimeError(f"Only {len(used)} poses saw the ball (need {min_poses}).")
    T, rms = calibration.solve_rb_to_body(np.array(ball_body), ball_world, T_world_rb)
    data = {"ball_body": np.array(ball_body), "ball_world": ball_world,
            "T_world_rb": np.array(T_world_rb), "candidates": candidates}
    if save_path is not None:
        save()
        with np.load(save_path, allow_pickle=True) as z:
            np.savez(save_path, **dict(z), ball_world=ball_world, T_rb_body=T, rms=rms)
    if verbose:
        ball_rb = [transform_points(inv_T(Tw), bw[None])[0] for Tw, bw in zip(T_world_rb, ball_world)]
        for k, bb, br, bw in zip(used, ball_body, ball_rb, ball_world):
            err = np.linalg.norm(transform_points(T, np.array(bb)[None])[0] - br)
            print(f"  pose {k + 1}: ball world {np.round(bw, 3)}, error {err * 1000:.0f} mm")
    return T, rms, data


def _stable_points(frame_points, radius=0.02, min_fraction=0.5):
    """Cluster the unlabeled markers of a static scene; keep clusters seen in
    at least min_fraction of the frames. Returns their median positions."""
    clusters = []
    for pts in frame_points:
        for p in pts:
            for c in clusters:
                if np.linalg.norm(c[0] - p) < radius:
                    c.append(p)
                    break
            else:
                clusters.append([p])
    n = max(len(frame_points), 1)
    return [np.median(c, axis=0) for c in clusters if len(c) >= min_fraction * n]


def _pick_ball(candidates, T_world_rb, static_radius=0.05):
    """The ball moves with the arm between poses; stray reflections do not.
    Drop candidates that sit at the same world position (within 5 cm: a
    marker on Spot's back drifts a few cm as the body sways) in most poses, then
    take the one closest to Spot's body markers."""
    picked = []
    for k, cands in enumerate(candidates):
        others = [c for j, cs in enumerate(candidates) if j != k for c in cs]
        moving = [c for c in cands
                  if sum(np.linalg.norm(o - c) < static_radius for o in others) < len(candidates) / 2]
        if not moving:
            picked.append(np.full(3, np.nan))  # only static points: the ball was not seen
            continue
        origin = T_world_rb[k][:3, 3]
        picked.append(min(moving, key=lambda c: np.linalg.norm(c - origin)))
    return np.array(picked)
