"""Minimal NatNet client (OptiTrack Motive streaming), no SDK needed.

Packet layouts follow OptiTrack's official Python sample (NatNetClient.py,
NatNet 3.x / 4.0 / 4.1+). Only what this project needs is kept: rigid bodies
(Spot's body) and markers (the ball, which is an unlabeled marker).

The encoder at the bottom builds the same packets; it is only used by the
fake Motive server in twin/fake_world.py to test this client without Motive.

Lab setup (see spot-docs 1_sensor_streaming.md): Motive PC 10.0.0.112 on the
SpotRoom network, Transmission type = unicast. Your PC must be on SpotRoom too.
"""

import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import mat_to_quat, quat_to_mat

NAT_CONNECT, NAT_SERVERINFO, NAT_REQUEST, NAT_RESPONSE = 0, 1, 2, 3
NAT_REQUEST_MODELDEF, NAT_MODELDEF, NAT_REQUEST_FRAMEOFDATA, NAT_FRAMEOFDATA = 4, 5, 6, 7
NAT_MESSAGESTRING, NAT_DISCONNECT, NAT_KEEPALIVE = 8, 9, 10

V3 = struct.Struct("<fff")
QUAT = struct.Struct("<ffff")
F32 = struct.Struct("<f")
F64 = struct.Struct("<d")
I32 = struct.Struct("<i")
I64 = struct.Struct("<q")
I16 = struct.Struct("<h")

# Motive streams Y-up by default; this rotation turns Y-up into Z-up
# (x stays, new y = -old z, new z = old y). Right-handed both ways.
Y_UP_TO_Z_UP = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


@dataclass
class RigidBody:
    id: int
    pos: np.ndarray
    quat: np.ndarray       # x, y, z, w
    error: float
    valid: bool


@dataclass
class Marker:
    pos: np.ndarray
    size: float
    model_id: int          # 0 = unlabeled (not part of any rigid body)
    marker_id: int
    occluded: bool


@dataclass
class MocapFrame:
    t_local: float         # time.time() when the packet arrived on this PC
    frame_number: int
    timestamp: float       # Motive's own clock (s)
    rigid_bodies: dict = field(default_factory=dict)   # id -> RigidBody
    markers: list = field(default_factory=list)        # labeled-marker section (all markers, NatNet 3+)
    other_markers: list = field(default_factory=list)  # legacy unlabeled section (positions only)

    def unlabeled(self):
        """Positions of markers that belong to no rigid body: ball candidates."""
        # Unlabeled markers usually have model_id 0, but Motive can also stream
        # them under another asset id (seen 2026-11-13: model_id 4). So: every
        # marker that is not part of a streamed rigid body.
        pts = [m.pos for m in self.markers if m.model_id not in self.rigid_bodies and not m.occluded]
        if not pts and not self.markers:  # very old servers: only the legacy list
            pts = list(self.other_markers)
        return pts


# ---------------------------------------------------------------- parsing ---

class _Reader:
    def __init__(self, data):
        self.d = memoryview(data)
        self.o = 0

    def take(self, st):
        v = st.unpack_from(self.d, self.o)
        self.o += st.size
        return v if len(v) > 1 else v[0]

    def string(self):
        end = bytes(self.d[self.o:]).index(b"\0")
        s = bytes(self.d[self.o:self.o + end]).decode("utf-8", "replace")
        self.o += end + 1
        return s


def _has_sizes(major, minor):
    # NatNet 4.1+ puts a byte count after every section count.
    return (major == 4 and minor > 0) or major > 4


def _rigid_body(r, major, minor):
    rb_id = r.take(I32)
    pos = np.array(r.take(V3))
    quat = np.array(r.take(QUAT))
    if major >= 3:
        err = r.take(F32)
        valid = bool(r.take(I16) & 0x01)
        return RigidBody(rb_id, pos, quat, err, valid)
    # NatNet 2.x: marker arrays inside the rigid body
    n = r.take(I32)
    r.o += 12 * n
    if major >= 2:
        r.o += 4 * n + 4 * n
        err = r.take(F32)
        valid = True
        if minor >= 6:
            valid = bool(r.take(I16) & 0x01)
        return RigidBody(rb_id, pos, quat, err, valid)
    return RigidBody(rb_id, pos, quat, 0.0, True)


def parse_frame(payload, major, minor, t_local=None):
    """Parse the payload of a NAT_FRAMEOFDATA packet (after the 4-byte header)."""
    r = _Reader(payload)
    sizes = _has_sizes(major, minor)
    frame = MocapFrame(time.time() if t_local is None else t_local, r.take(I32), 0.0)

    # Marker sets (named groups of marker positions; skipped)
    n = r.take(I32)
    if sizes:
        r.take(I32)
    for _ in range(n):
        r.string()
        r.o += 12 * r.take(I32)

    # Legacy "other" (unlabeled) markers
    n = r.take(I32)
    if sizes:
        r.take(I32)
    frame.other_markers = [np.array(r.take(V3)) for _ in range(n)]

    # Rigid bodies
    n = r.take(I32)
    if sizes:
        r.take(I32)
    for _ in range(n):
        rb = _rigid_body(r, major, minor)
        frame.rigid_bodies[rb.id] = rb

    # Skeletons (2.1+): id, rigid body count, rigid bodies
    if (major == 2 and minor > 0) or major > 2:
        n = r.take(I32)
        if sizes:
            r.take(I32)
        for _ in range(n):
            r.take(I32)
            for _ in range(r.take(I32)):
                _rigid_body(r, major, minor)

    # Assets (4.1+)
    if sizes:
        n = r.take(I32)
        r.take(I32)
        for _ in range(n):
            r.take(I32)
            r.o += 38 * r.take(I32)   # id, pos, quat, error, params
            r.o += 26 * r.take(I32)   # id, pos, size, params, residual

    # Labeled markers (2.3+): in NatNet 3+ this lists all markers, with
    # model id 0 for unlabeled ones.
    if (major == 2 and minor > 3) or major > 2:
        n = r.take(I32)
        if sizes:
            r.take(I32)
        for _ in range(n):
            raw_id = r.take(I32)
            pos = np.array(r.take(V3))
            size = r.take(F32)
            param = r.take(I16) if ((major == 2 and minor >= 6) or major > 2) else 0
            if major >= 3:
                r.take(F32)  # residual
            frame.markers.append(Marker(pos, size, raw_id >> 16, raw_id & 0xFFFF, bool(param & 0x01)))

    # Force plates (2.9+) and devices (2.11+): channel data, skipped
    for present in ((major == 2 and minor >= 9) or major > 2, (major == 2 and minor >= 11) or major > 2):
        if not present:
            continue
        n = r.take(I32)
        if sizes:
            r.take(I32)
        for _ in range(n):
            r.take(I32)
            for _ in range(r.take(I32)):
                r.o += 4 * r.take(I32)

    # Suffix: timecode, timecode sub, timestamp, ...
    r.take(I32)
    r.take(I32)
    frame.timestamp = r.take(F64) if (major >= 3 or major == 0 or (major == 2 and minor >= 7)) else r.take(F32)
    return frame


def to_z_up(frame):
    """Convert a Y-up frame to Z-up in place."""
    R = Y_UP_TO_Z_UP
    for rb in frame.rigid_bodies.values():
        rb.pos = R @ rb.pos
        rb.quat = mat_to_quat(R @ quat_to_mat(rb.quat))
    for m in frame.markers:
        m.pos = R @ m.pos
    frame.other_markers = [R @ p for p in frame.other_markers]
    return frame


# ----------------------------------------------------------------- client ---

def _request(command, payload=b""):
    return struct.pack("<hH", command, len(payload) & 0xFFFF) + payload


def connect_payload(version=(4, 1, 0, 0)):
    """NAT_CONNECT body: 'Ping', zero padding, requested NatNet version."""
    body = bytearray(270)
    body[0:4] = b"Ping"
    body[265:269] = bytes(version)
    return bytes(body) + b"\0"


class NatNetClient:
    """Receives frames in background threads.

    on_frame(frame) is called for every frame (from a background thread);
    latest() returns the most recent one; recent frames are kept in a buffer.
    up_axis: what Motive streams ('y' default in Motive, or 'z'). Frames
    handed out are always Z-up.
    """

    def __init__(self, server_ip, local_ip, multicast=False, multicast_address="239.255.42.99",
                 command_port=1510, data_port=1511, up_axis="z", on_frame=None, buffer_size=10000):
        self.server_ip, self.local_ip = server_ip, local_ip
        self.multicast, self.multicast_address = multicast, multicast_address
        self.command_port, self.data_port = command_port, data_port
        self.up_axis = up_axis
        self.on_frame = on_frame
        self.frames = deque(maxlen=buffer_size)
        self.version = None          # NatNet version reported by the server
        self.server_name = None
        self._stop = threading.Event()
        self._threads = []
        self._lock = threading.Lock()

    # -- setup --

    def start(self, timeout=3.0):
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.cmd_sock.bind((self.local_ip, 0))
        self.cmd_sock.settimeout(0.5)
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.multicast:
            self.data_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                                      socket.inet_aton(self.multicast_address) + socket.inet_aton(self.local_ip))
            self.data_sock.bind((self.local_ip, self.data_port))
        else:
            # Unicast frames may arrive on the command socket or on the data
            # port depending on the Motive version; listen on both.
            try:
                self.data_sock.bind((self.local_ip, self.data_port))
            except OSError:
                self.data_sock.bind((self.local_ip, 0))
        self.data_sock.settimeout(0.5)

        for sock in (self.cmd_sock, self.data_sock):
            th = threading.Thread(target=self._recv_loop, args=(sock,), daemon=True)
            th.start()
            self._threads.append(th)
        if not self.multicast:
            th = threading.Thread(target=self._keepalive_loop, daemon=True)
            th.start()
            self._threads.append(th)

        self.cmd_sock.sendto(_request(NAT_CONNECT, connect_payload()), (self.server_ip, self.command_port))
        deadline = time.time() + timeout
        while self.version is None and time.time() < deadline:
            time.sleep(0.01)
        if self.version is None:
            self.stop()
            raise TimeoutError(
                f"No answer from Motive at {self.server_ip}:{self.command_port}. Check: same network "
                f"(SpotRoom), Motive streaming enabled, local interface / unicast settings, firewall (UDP).")
        return self

    def stop(self):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=1.0)
        for sock in (getattr(self, "cmd_sock", None), getattr(self, "data_sock", None)):
            if sock is not None:
                sock.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- receiving --

    def _keepalive_loop(self):
        while not self._stop.wait(1.0):
            try:
                self.cmd_sock.sendto(_request(NAT_KEEPALIVE), (self.server_ip, self.command_port))
            except OSError:
                pass

    def _recv_loop(self, sock):
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(128 * 1024)
            except socket.timeout:
                continue
            except OSError:
                return
            t = time.time()
            if len(data) < 4:
                continue
            msg_id = I16.unpack_from(data, 0)[0]
            if msg_id == NAT_SERVERINFO:
                self._server_info(data[4:])
            elif msg_id == NAT_FRAMEOFDATA and self.version is not None:
                try:
                    frame = parse_frame(data[4:], self.version[0], self.version[1], t)
                except (struct.error, ValueError) as err:
                    print(f"[natnet] bad frame skipped: {err}")
                    continue
                if self.up_axis == "y":
                    to_z_up(frame)
                with self._lock:
                    self.frames.append(frame)
                if self.on_frame is not None:
                    self.on_frame(frame)

    def _server_info(self, payload):
        self.server_name = bytes(payload[:256]).split(b"\0", 1)[0].decode("utf-8", "replace")
        self.version = tuple(payload[260:264])

    def latest(self):
        with self._lock:
            return self.frames[-1] if self.frames else None

    def frames_between(self, t0, t1):
        with self._lock:
            return [f for f in self.frames if t0 <= f.t_local <= t1]


# --------------------------------------------- encoder (fake server only) ---

def server_info_packet(name="Motive (fake)", server_version=(3, 1, 0, 0), natnet_version=(4, 1, 0, 0)):
    body = name.encode()[:255].ljust(256, b"\0") + bytes(server_version) + bytes(natnet_version)
    return _request(NAT_SERVERINFO, body)


def _section(count, body, sizes):
    return I32.pack(count) + (I32.pack(len(body)) if sizes else b"") + body


def encode_frame(frame_number, rigid_bodies, markers, major, minor, timestamp=0.0, legacy_unlabeled=True):
    """Build a NAT_FRAMEOFDATA packet (NatNet 3+).
    rigid_bodies: list of (id, pos, quat_xyzw); markers: list of (pos, size, model_id, marker_id)."""
    sizes = _has_sizes(major, minor)
    out = I32.pack(frame_number)
    out += _section(0, b"", sizes)                                     # marker sets
    unl = [m for m in markers if m[2] == 0] if legacy_unlabeled else []
    out += _section(len(unl), b"".join(V3.pack(*m[0]) for m in unl), sizes)
    rb_body = b"".join(I32.pack(i) + V3.pack(*p) + QUAT.pack(*q) + F32.pack(0.0005) + I16.pack(1)
                       for i, p, q in rigid_bodies)
    out += _section(len(rigid_bodies), rb_body, sizes)
    out += _section(0, b"", sizes)                                     # skeletons
    if sizes:
        out += _section(0, b"", sizes)                                 # assets
    lm_body = b"".join(I32.pack((model << 16) | mid) + V3.pack(*p) + F32.pack(size) + I16.pack(0)
                       + F32.pack(0.0) for p, size, model, mid in markers)
    out += _section(len(markers), lm_body, sizes)
    out += _section(0, b"", sizes)                                     # force plates
    out += _section(0, b"", sizes)                                     # devices
    out += I32.pack(0) + I32.pack(0) + F64.pack(timestamp)             # timecode, sub, timestamp
    out += I64.pack(0) * 3                                             # camera / receive / transmit stamps
    if sizes:
        out += I32.pack(0) * 2                                         # precision timestamp (4.1+)
    out += I16.pack(0)                                                 # params
    return _request(NAT_FRAMEOFDATA, out)
