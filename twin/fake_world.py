"""Fake Motive + fake Spot, to test the live pipeline without the lab.

FakeWorld   - the "truth": the PyBullet simulation placed somewhere in a room,
              with OptiTrack markers on Spot's body and a ball.
FakeMotive  - UDP server speaking NatNet (same packets as Motive), streaming
              FakeWorld as OptiTrack would: rigid body + labeled markers for
              the body, the ball as an unlabeled marker, noise, dropouts and
              stray reflections.
FakeRobot   - same interface as live.SpotRobotInterface; plays the simulated
              throw in real time and reports arm joints like robot_state.

Use: python run_twin.py fake-live   (or tests/test_twin_live.py)
"""

import socket
import threading
import time

import numpy as np

from . import natnet
from .geometry import inv_T, make_T, mat_to_quat, rot_z, T_to_pos_quat, transform_points
from .kinematics import ArmFK

# Marker positions on Spot's back, in the rigid-body frame.
BODY_MARKERS = np.array([[0.10, 0.05, 0.0], [-0.10, 0.05, 0.0], [0.0, -0.08, 0.0], [0.05, 0.0, 0.03]])


def room_pose(x, y, yaw_deg):
    T = np.eye(4)
    T[:3, :3] = rot_z(np.radians(yaw_deg))
    T[:3, 3] = (x, y, 0.0)
    return T


class FakeWorld:
    def __init__(self, sim_source, start_pose, T_room_world=None, T_rb_body=None, rb_id=1, up_axis="z",
                 distractors=((0.5, 2.0, 1.2), (-1.0, -0.5, 0.02)), noise_m=0.0005, dropout=0.02, seed=0):
        self.sim = sim_source
        # Where the simulated robot stands in the OptiTrack room (floor stays z = 0).
        self.T_room_world = room_pose(1.5, -0.8, 35.0) if T_room_world is None else T_room_world
        # Where Motive put the rigid-body pivot relative to Spot's body: unknown
        # to the pipeline, recovered by calibration.
        if T_rb_body is None:
            T_body_rb = make_T((-0.12, 0.03, 0.11), mat_to_quat(rot_z(np.radians(8))))
            T_rb_body = inv_T(T_body_rb)
        self.T_rb_body = T_rb_body
        self.rb_id = rb_id
        self.up_axis = up_axis
        self.distractors = np.array(distractors, float).reshape(-1, 3)
        self.noise_m, self.dropout = noise_m, dropout
        self.rng = np.random.default_rng(seed)
        self.fk = ArmFK()
        self.fk_lock = threading.Lock()
        self.lock = threading.Lock()
        self.q = np.array(start_pose, float)
        self.playback = None
        self.last_result = None
        # Standing body pose: settle the simulation once with the arm still.
        still = (np.array([0.0, 0.3]), np.array([start_pose, start_pose]), np.zeros((2, 6)))
        res = sim_source.simulate(0.3, 0.99, trajectory=still)
        self.T_world_body_standing = make_T(res["log"]["body_pos"][0], res["log"]["body_orn"][0])

    # -- truth at a given local time --

    def state(self, t):
        """(T_world_body, q, ball_world or None) at local time t, in the sim world."""
        with self.lock:
            pb = self.playback
            q_static = self.q.copy()
        if pb is None or t < pb["t0"]:
            T_wb = self.T_world_body_standing
            with self.fk_lock:  # one PyBullet client, used from several threads
                ball = self.fk.grasp_point(q_static, *T_to_pos_quat(T_wb))
            return T_wb, q_static, ball
        k = int(np.clip(np.searchsorted(pb["t"], t - pb["t0"]), 0, len(pb["t"]) - 1))
        T_wb = make_T(pb["body_pos"][k], pb["body_orn"][k])
        return T_wb, pb["q"][k], pb["ball"][k]

    def start_playback(self, res, t0):
        log = res["log"]
        with self.lock:
            self.playback = {"t0": t0, "t": np.array(log["t"]), "q": np.array(log["q"]),
                             "ball": np.array(log["ball"]), "body_pos": np.array(log["body_pos"]),
                             "body_orn": np.array(log["body_orn"])}
            self.last_result = res

    def stop_playback(self, q):
        with self.lock:
            self.playback = None
            self.q = np.array(q, float)

    # -- what OptiTrack sees --

    def mocap(self, t):
        """Rigid bodies and markers as Motive would stream them (room frame)."""
        T_wb, _, ball = self.state(t)
        T_room_body = self.T_room_world @ T_wb
        T_room_rb = T_room_body @ inv_T(self.T_rb_body)
        noise = lambda n: self.rng.normal(0, self.noise_m, (n, 3))  # noqa: E731
        rb_pos, rb_quat = T_to_pos_quat(T_room_rb)
        rb_markers = transform_points(T_room_rb, BODY_MARKERS) + noise(len(BODY_MARKERS))
        markers = [(p, 0.014, self.rb_id, i + 1) for i, p in enumerate(rb_markers)]
        if ball is not None and self.rng.random() > self.dropout:
            ball_room = transform_points(self.T_room_world, np.asarray(ball)[None])[0] + noise(1)[0]
            markers.append((ball_room, 0.067, 0, 0))
        for i, d in enumerate(self.distractors):
            markers.append((d + noise(1)[0], 0.01, 0, 100 + i))
        rbs = [(self.rb_id, rb_pos + noise(1)[0] * 0.3, rb_quat)]
        if self.up_axis == "y":  # Motive default: stream Y-up
            R = natnet.Y_UP_TO_Z_UP.T
            rbs = [(i, R @ p, mat_to_quat(R @ make_T(p, q)[:3, :3])) for i, p, q in rbs]
            markers = [(R @ p, s, m, k) for p, s, m, k in markers]
        return rbs, markers


class FakeMotive:
    """NatNet server on UDP (unicast). Streams FakeWorld at rate_hz."""

    def __init__(self, world, ip="127.0.0.1", command_port=1510, natnet_version=(4, 1, 0, 0), rate_hz=120.0):
        self.world = world
        self.ip, self.command_port = ip, command_port
        self.version = natnet_version
        self.rate_hz = rate_hz
        self.clients = {}
        self._stop = threading.Event()

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.bind((self.ip, self.command_port))
        self.sock.settimeout(0.05)
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._serve, self._stream)]
        for th in self.threads:
            th.start()
        return self

    def stop(self):
        self._stop.set()
        for th in self.threads:
            th.join(timeout=1.0)
        self.sock.close()

    def _serve(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
            except (socket.timeout, ConnectionResetError):
                continue
            except OSError:
                return
            msg = natnet.I16.unpack_from(data, 0)[0]
            if msg == natnet.NAT_CONNECT:
                self.clients[addr] = time.time()
                self.sock.sendto(natnet.server_info_packet(natnet_version=self.version), addr)
            elif msg == natnet.NAT_KEEPALIVE:
                self.clients[addr] = time.time()

    def _stream(self):
        frame_no = 0
        major, minor = self.version[:2]
        next_t = time.time()
        while not self._stop.is_set():
            next_t += 1.0 / self.rate_hz
            time.sleep(max(0.0, next_t - time.time()))
            if not self.clients:
                continue
            t = time.time()
            rbs, markers = self.world.mocap(t)
            pkt = natnet.encode_frame(frame_no, rbs, markers, major, minor, timestamp=t)
            frame_no += 1
            for addr, seen in list(self.clients.items()):
                if t - seen > 5.0:  # no keep-alive: client gone
                    del self.clients[addr]
                    continue
                try:
                    self.sock.sendto(pkt, addr)
                except OSError:
                    pass


class FakeRobot:
    """Same interface as live.SpotRobotInterface, driven by FakeWorld."""

    def __init__(self, world, joint_rate_hz=60.0):
        self.world = world
        self.joint_rate_hz = joint_rate_hz

    def joint_state(self):
        t = time.time()
        return t, np.array(self.world.state(t)[1])

    def move_arm(self, q, seconds=4.0):
        time.sleep(0.1)  # the real arm takes `seconds`; the fake one is instant
        self.world.stop_playback(q)

    def run_throw(self, duration, release_fraction, trajectory=None):
        times, positions, velocities = trajectory
        self.world.stop_playback(positions[0])
        res = self.world.sim.simulate(duration, release_fraction, trajectory=trajectory)
        t0 = time.time() + 0.5
        self.world.start_playback(res, t0)
        log_t, log_q = [], []
        while time.time() < t0 + duration + 0.5:
            t, q = self.joint_state()
            log_t.append(t)
            log_q.append(q)
            time.sleep(1.0 / self.joint_rate_hz)
        return {"start_time": t0, "release_cmd_time": t0 + res["release_cmd_t"],
                "joint_t": np.array(log_t), "q": np.array(log_q)}
