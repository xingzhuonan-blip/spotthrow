"""Fixed cubic throw for the dataset: only two settings are free.

Every joint goes from START_POSE to END_POSE along the same cubic curve
(zero speed at both ends). The poses give the same overhead arc as the fast
throws (see below), so the path is always the same; only the timing changes:

    swing_time    s, start pose -> end pose
    release_time  s after the swing starts, gripper open command
                  (the ball leaves ~0.1-0.3 s later, the gripper's own delay)
"""

from dataclasses import asdict, dataclass, fields

import numpy as np

from throw_trajectory import MAX_JOINT_ACC, MAX_JOINT_VEL, check_trajectory, cubic_joint_trajectory

# Overhead arc like the fast throws run 
# elbow straight (el0 0.05), shoulder sh1 and wrist wr0 swing together, sh0 = 0 so
# the throw goes straight forward. sh1 starts at -2.2 (box limit is -2.3, margin 0.05).
START_POSE = [0.0, -2.2, 0.05, 0.0, -1.43, 1.57]
END_POSE = [0.0, -0.52, 0.05, 0.0, 0.30, 1.57]


@dataclass
class CubicThrow:
    swing_time: float = 0.60
    release_time: float = 0.25

    HELP = {
        "swing_time": "s, start pose -> end pose (shorter = faster throw)",
        "release_time": "s after the swing starts, gripper open command (ball leaves ~0.1-0.3 s later)",
    }

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]

    def as_dict(self):
        return asdict(self)

    def build(self, dt=0.01):
        """-> (duration, release_fraction, (times, positions, velocities))."""
        T = self.swing_time
        return T, float(np.clip(self.release_time / T, 0.0, 1.0)), cubic_joint_trajectory(START_POSE, END_POSE, T, dt)

    def problems(self, max_vel=MAX_JOINT_VEL, max_acc=MAX_JOINT_ACC):
        """Empty list = OK for the robot."""
        if self.swing_time < 0.1:
            return ["swing_time too short"]
        if not 0.0 <= self.release_time <= self.swing_time:
            return ["release_time must be between 0 and swing_time"]
        return list(check_trajectory(*self.build()[2], max_vel, max_acc))

    def to_plan(self, gripper_delay=0.25, max_vel=MAX_JOINT_VEL, max_acc=MAX_JOINT_ACC):
        """Plan-file dict (cubic format, read by twin.planner.load_plan)."""
        T, rf, _ = self.build()
        return {"target": None, "duration": T, "release_fraction": rf,
                "start_pose": START_POSE, "end_pose": END_POSE, "sim_landing": None,
                "gripper_delay": gripper_delay, "space": {"max_vel": max_vel, "max_acc": max_acc}}
