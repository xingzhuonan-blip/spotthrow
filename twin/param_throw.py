"""A fixed forward throw whose timing the user sets by hand (for the dataset).

The arm swings straight forward (sh0 = 0) with the same profile as the fast
throw:

    accelerate (accel_time) -> constant speed (coast_time) -> brake (brake_time)

Only sh1 (shoulder) and wr0 (wrist) move; el0 is held. The pose where braking
starts is fixed (brake_sh1 and the wrist pose below), so changing the times or
speeds changes the wind-up, not where the arm stops speeding.

The gripper open COMMAND is sent at open_time, counted from the start of the
constant-speed part (negative = still accelerating). The ball leaves about
0.1-0.3 s later (the gripper's own delay).

Defaults: the ball must leave while the arm is still at constant speed, not
while it brakes (braking with the ball loose in a half-open gripper gave
1.47-1.61 m spread for identical throws, 2026-11-05). Ball free = open_time +
gripper delay after the coast starts; with open_time 0.15 and coast 0.45 that
holds for delays up to 0.29 s (measured with the slow open). The release
command now opens the gripper fast (spot_throw_safe.FAST_OPEN_VEL).

A longer coast or a higher speed winds up further back: if the limits check
fails on sh1, make brake_sh1 less negative.
"""

from dataclasses import asdict, dataclass, fields

import numpy as np

from throw_trajectory import check_trajectory

EL0, EL1, WR1 = 0.05, 0.0, 1.57
WR0_AT_BRAKE = -0.70  # wrist pitch where braking starts


@dataclass
class ParamThrow:
    accel_time: float = 0.35   # s, speeding up from rest
    coast_time: float = 0.45   # s, constant speed
    brake_time: float = 0.35   # s, slowing down to rest
    speed_sh1: float = 2.12    # rad/s, shoulder speed while coasting
    speed_wr0: float = 0.58    # rad/s, wrist speed while coasting
    brake_sh1: float = -0.90   # rad, shoulder angle where braking starts
    open_time: float = 0.15    # s after the coast starts: gripper open command

    HELP = {
        "accel_time": "s, arm speeds up from rest",
        "coast_time": "s, arm at constant speed",
        "brake_time": "s, arm slows down to rest",
        "speed_sh1": "rad/s, shoulder speed (bigger = faster throw)",
        "speed_wr0": "rad/s, wrist flick speed",
        "brake_sh1": "rad, shoulder angle where braking starts (more negative = earlier/higher)",
        "open_time": "s after the coast starts, gripper open command (ball leaves ~0.1-0.3 s later)",
    }

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]

    def as_dict(self):
        return asdict(self)

    def build(self, dt=0.01):
        """-> (duration, release_fraction, (times, positions, velocities))."""
        ta, tc, tb = self.accel_time, self.coast_time, self.brake_time
        w = np.array([0.0, self.speed_sh1, 0.0, 0.0, self.speed_wr0, 0.0])
        q_brake = np.array([0.0, self.brake_sh1, EL0, EL1, WR0_AT_BRAKE, WR1])
        T = ta + tc + tb
        times = np.linspace(0.0, T, int(round(T / dt)) + 1)
        s = np.zeros_like(times)  # speed fraction 0..1
        d = np.zeros_like(times)  # distance in "full speed seconds"
        for i, t in enumerate(times):
            if t < ta:
                s[i] = 0.5 * (1 - np.cos(np.pi * t / ta))
                d[i] = 0.5 * (t - ta / np.pi * np.sin(np.pi * t / ta))
            elif t < ta + tc:
                s[i] = 1.0
                d[i] = ta / 2 + (t - ta)
            else:
                u = min(t - ta - tc, tb)
                s[i] = 0.5 * (1 + np.cos(np.pi * u / tb))
                d[i] = ta / 2 + tc + 0.5 * (u + tb / np.pi * np.sin(np.pi * u / tb))
        q0 = q_brake - w * (ta / 2 + tc)  # so that braking starts at q_brake
        positions = q0 + np.outer(d, w)
        velocities = np.outer(s, w)
        t_open = ta + self.open_time
        return T, float(np.clip(t_open / T, 0.0, 1.0)), (times, positions, velocities)

    def problems(self, max_vel=2.5, max_acc=10.0):
        """Empty list = OK for the robot (same joint checks as the throw itself)."""
        out = []
        for k in ("accel_time", "coast_time", "brake_time"):
            if getattr(self, k) < (0.0 if k == "coast_time" else 0.1):
                out.append(f"{k} too short")
        t_open = self.accel_time + self.open_time
        if not 0.0 <= t_open <= self.accel_time + self.coast_time + self.brake_time:
            out.append("open_time is outside the motion")
        if out:
            return out
        T, rf, traj = self.build()
        return out + list(check_trajectory(*traj, max_vel, max_acc))

    def to_plan(self, gripper_delay=0.25, max_vel=2.5, max_acc=10.0):
        """Plan-file dict (kind "param"), read by twin.planner.load_plan."""
        return {"kind": "param", "params": self.as_dict(), "target": None,
                "gripper_delay": gripper_delay, "space": {"max_vel": max_vel, "max_acc": max_acc}}


class _Adapter:
    """Lets FastThrowPlanner's checks (body clearance, re-hit) and physics
    simulation run on a ParamThrow."""

    def __init__(self, pt, gripper_delay):
        self.pt, self.gripper_delay = pt, gripper_delay

    def build(self, x=None, dt=0.01):
        return self.pt.build(dt)


def check(pt, gripper_delay=0.25, fp=None):
    """Body clearance and a physics-simulated landing (robot frame).
    Returns dict(clearance m, landing (x, y) or None). Takes a few seconds."""
    from .fast_throw import FastThrowPlanner, FastThrowSpace
    own = fp is None
    if own:
        fp = FastThrowPlanner(FastThrowSpace(gripper_delay=gripper_delay))
    saved = fp.space
    try:
        fp.space = _Adapter(pt, gripper_delay)
        clearance = fp.min_clearance(None)
        fp.base.gripper_delay = fp.base.sim.gripper_delay = gripper_delay
        land, _ = fp.simulate(None)
        return {"clearance": clearance, "landing": None if land is None else land[:2]}
    finally:
        fp.space = saved
        if own:
            fp.close()
