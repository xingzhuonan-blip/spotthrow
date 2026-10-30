"""Forward kinematics of the Spot arm using the PyBullet model (headless)."""

from pathlib import Path

import numpy as np
import pybullet as p

from .geometry import inv_T, make_T, T_to_pos_quat

ROOT = Path(__file__).resolve().parent.parent
URDF = str(ROOT / "fixed_spot" / "spot_fixed_base.urdf")
ARM_JOINTS = [18, 19, 21, 22, 23, 24]   # SDK order arm0.sh0 ... arm0.wr1
BODY_LINK = 2                           # URDF 'body' link = Spot's body frame
WRIST_LINK = 24
GRASP_OFFSET = (0.19, 0.0, -0.01)       # ball centre between the jaws, in the wr1 link frame
LEGS = (0.1, 0.9, -1.5, -0.1, 0.9, -1.5, 0.1, 1.1, -1.5, -0.1, 1.1, -1.5)


def root_to_body(cid, robot):
    """Fixed transform from the URDF root to the 'body' link."""
    p.resetBasePositionAndOrientation(robot, (0, 0, 0), (0, 0, 0, 1), physicsClientId=cid)
    pos, orn = p.getLinkState(robot, BODY_LINK, computeForwardKinematics=True, physicsClientId=cid)[4:6]
    return make_T(pos, orn)


def place_body(cid, robot, T_root_body, body_pos, body_orn):
    """Put the model so that its body link sits at (body_pos, body_orn)."""
    pos, orn = T_to_pos_quat(make_T(body_pos, body_orn) @ inv_T(T_root_body))
    p.resetBasePositionAndOrientation(robot, pos, orn, physicsClientId=cid)


class ArmFK:
    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        self.robot = p.loadURDF(URDF, physicsClientId=self.cid)
        for i, q in enumerate(LEGS):
            p.resetJointState(self.robot, 6 + i, q, physicsClientId=self.cid)
        self.T_root_body = root_to_body(self.cid, self.robot)

    def grasp_point(self, q, body_pos=(0, 0, 0), body_orn=(0, 0, 0, 1)):
        """Gripper centre for arm joints q, with the body at (body_pos, body_orn).
        With the defaults the result is in the body frame."""
        place_body(self.cid, self.robot, self.T_root_body, body_pos, body_orn)
        for j, qj in zip(ARM_JOINTS, q):
            p.resetJointState(self.robot, j, qj, physicsClientId=self.cid)
        pos, orn = p.getLinkState(self.robot, WRIST_LINK, computeForwardKinematics=True,
                                  physicsClientId=self.cid)[4:6]
        return np.array(p.multiplyTransforms(pos, orn, GRASP_OFFSET, (0, 0, 0, 1))[0])

    def close(self):
        p.disconnect(self.cid)


def gripper_path(rec, fk=None):
    """Gripper centre in the robot frame at every ball sample time, shape (M, 3)."""
    own = fk is None
    fk = fk or ArmFK()
    try:
        return np.array([fk.grasp_point(rec.q_at(t), *rec.body_at(t)) for t in rec.ball_t])
    finally:
        if own:
            fk.close()
