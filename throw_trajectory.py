"""Joint-space throw trajectories that do not depend on the Spot SDK, so the
simulation, the digital twin and the robot scripts can all share them.

Joint order everywhere: SDK arm0.sh0, sh1, el0, el1, wr0, wr1.
"""

import numpy as np

ARM_JOINT_NAMES = ("arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1")
JOINT_LIMITS = np.array([
    (-2.618, 3.141),   # sh0
    (-3.141, 0.524),   # sh1 (hardware); lower end tightened by SH1_MIN below
    (0.0, 3.141),      # el0
    (-2.793, 2.793),   # el1
    (-1.833, 1.833),   # wr0
    (-2.880, 2.880),   # wr1
])
JOINT_LIMIT_MARGIN = 0.05
# Keep the shoulder from swinging the arm back into the box mounted on Spot
# (the windup reached -2.7 rad and hit it). Every planner and the pre-throw
# check use JOINT_LIMITS, so this one number limits all of them.
SH1_MIN = -2.3
JOINT_LIMITS[1, 0] = SH1_MIN

# Joint speed / acceleration caps for throws (raised 2026-11-06 for faster
# throws; the gripper fast-open command already uses 10 rad/s, 100 rad/s^2).
MAX_JOINT_VEL = 10.0   # rad/s
MAX_JOINT_ACC = 100.0  # rad/s^2


def cubic_joint_trajectory(start, end, T, dt=0.01):
    """All joints move from start to end with the same cubic timing (zero
    velocity at both ends). Returns times (n,), positions (n, 6), velocities (n, 6)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    times = np.linspace(0.0, T, int(round(T / dt)) + 1)
    s = times / T
    d = end - start
    positions = start + np.outer(3 * s**2 - 2 * s**3, d)
    velocities = np.outer((6 * s - 6 * s**2) / T, d)
    return times, positions, velocities


def check_trajectory(times, positions, velocities, max_vel, max_acc):
    """Generic limit check for any sampled joint trajectory. Returns a list of
    problems (empty = OK). Acceleration is estimated from the velocity samples."""
    times, positions, velocities = map(np.asarray, (times, positions, velocities))
    problems = []
    lo = JOINT_LIMITS[:, 0] + JOINT_LIMIT_MARGIN
    hi = JOINT_LIMITS[:, 1] - JOINT_LIMIT_MARGIN
    for j, name in enumerate(ARM_JOINT_NAMES):
        if positions[:, j].min() < lo[j] or positions[:, j].max() > hi[j]:
            problems.append(f"{name} leaves its safe range [{lo[j]:.3f}, {hi[j]:.3f}]: "
                            f"[{positions[:, j].min():.3f}, {positions[:, j].max():.3f}]")
        v = np.abs(velocities[:, j]).max()
        if v > max_vel:
            problems.append(f"{name} peak velocity {v:.2f} rad/s > {max_vel:.2f}")
        a = np.abs(np.gradient(velocities[:, j], times)).max()
        if a > max_acc:
            problems.append(f"{name} peak acceleration {a:.1f} rad/s^2 > {max_acc:.1f}")
    return problems
