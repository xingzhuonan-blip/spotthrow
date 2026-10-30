import argparse
import sys
import time
from typing import Optional

import numpy as np

import bosdyn.client.util
from bosdyn.api import robot_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    blocking_stand,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient


ARM_JOINT_NAMES = ("arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1")

# Start pose used by initial_move(). el0 starts slightly inside its upper limit (pi).
START_POSE = [0, -2.5, 3.0, 0, np.pi/2, 0]
ELBOW_END = np.pi/2

# Finite limits passed to the SDK (rad/s, rad/s^2).
MAX_JOINT_VEL = 8.0
MAX_JOINT_ACC = 20.0


def elbow_traj(t:float, T:float=1):
    """Cubic from START_POSE[2] to ELBOW_END with zero start/end velocity."""
    t = min(max(t, 0.0), T)  # hold the end pose instead of extrapolating past T
    d = ELBOW_END - START_POSE[2]
    a0 = START_POSE[2]
    a1 = 0
    a2 = 3*d/(T**2)
    a3 = -2*d/(T**3)
    angle = a0 + a1*t+a2*t**2+a3*t**3
    vel = a1 + 2*a2*t+3*a3*t**2
    return angle, vel


def query_arm_joint_trajectory(t, T):
    """
    Given a time t in the trajectory, return the joint angles and velocities.
    This function can be modified to any arbitrary trajectory, as long as it
    can be sampled for any time t in seconds
    """
    theta, omega = elbow_traj(t, T)
    # Other joints hold the start pose (not 0), otherwise the first point
    # would make the shoulder/wrist jump.
    joint_positions = list(START_POSE)
    joint_velocities = [0, 0, 0, 0, 0, 0]
    joint_positions[2] = theta
    joint_velocities[2] = omega
    # Return the joint positions and velocities at time t in our trajectory
    return joint_positions, joint_velocities



def _make_arm_joint_cmd(joints, horizon_s: float):
    """Build a RobotCommand that drives the arm to 6 joint positions.

    Constructs the proto by setting fields on an empty RobotCommand, which
    avoids referencing nested proto class names that vary across SDK versions.
    The result can be merged with base / gripper commands via
    ``RobotCommandBuilder.build_synchro_command``.
    """
    cmd = robot_command_pb2.RobotCommand()
    point = cmd.synchronized_command.arm_command.arm_joint_move_command \
               .trajectory.points.add()
    point.position.sh0.value = float(joints[0])
    point.position.sh1.value = float(joints[1])
    point.position.el0.value = float(joints[2])
    point.position.el1.value = float(joints[3])
    point.position.wr0.value = float(joints[4])
    point.position.wr1.value = float(joints[5])
    point.time_since_reference.seconds = int(horizon_s)
    point.time_since_reference.nanos = int((horizon_s % 1) * 1e9)
    return cmd

class SpotThrows:
    def __init__(self, robot, use_software_estop=True):
        self._robot = robot
        self.use_software_estop = use_software_estop
        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        self._state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        try:
            self._estop_client = robot.ensure_client(EstopClient.default_service_name)
        except Exception:
            self._estop_client = None
        self._lease_keepalive: Optional[LeaseKeepAlive] = None
        self._estop_keepalive: Optional[EstopKeepAlive] = None
        self._mobility_params = RobotCommandBuilder.mobility_params()

    # ---- lifecycle ---------------------------------------------------

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False

    def acquire(self):
        self._robot.time_sync.wait_for_sync()
        if self.use_software_estop and self._estop_client is not None:
            ep = EstopEndpoint(self._estop_client, "spot_joint_replay", estop_timeout=9.0)
            ep.force_simple_setup()
            self._estop_keepalive = EstopKeepAlive(ep)
        self._lease_keepalive = LeaseKeepAlive(
            self._lease_client, must_acquire=True, return_at_exit=True,
        )

    def power_on(self):
        self._robot.power_on(timeout_sec=20)
        assert self._robot.is_powered_on(), "Failed to power on."

    def stand(self):
        blocking_stand(self._command_client, timeout_sec=10)

    def shutdown(self, sit_and_power_off: bool = True):
        try:
            if sit_and_power_off and self._robot.is_powered_on():
                self._command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
                time.sleep(1.0)
                self._robot.power_off(cut_immediately=False, timeout_sec=20)
        finally:
            for ka in (self._estop_keepalive, self._lease_keepalive):
                if ka is not None:
                    ka.shutdown()
            self._estop_keepalive = self._lease_keepalive = None

    def get_observation(self):
        state = self._state_client.get_robot_state()
        jmap = {js.name: js.position.value for js in state.kinematic_state.joint_states}
        arm_joints = np.array([jmap.get(n, 0.0) for n in ARM_JOINT_NAMES], dtype=float)
        gripper = state.manipulator_state.gripper_open_percentage / 100.0
        return arm_joints, gripper
    
    
    def initial_move(self):
        cmd = _make_arm_joint_cmd(START_POSE, 3.0)
        cmd_id = self._command_client.robot_command(cmd)
        block_until_arm_arrives(self._command_client, cmd_id)
        
    
    def throw(self, T=1):
        # The swing is short, so send it as ONE trajectory instead of streaming
        # segments. Sampled at 100 Hz, ending exactly at T.
        dt = 0.01
        times = list(np.linspace(0.0, T, int(round(T / dt)) + 1))
        positions = []
        velocities = []
        for t in times:
            pos, vel = query_arm_joint_trajectory(t, T)
            positions.append(pos)
            velocities.append(vel)

        # Start 0.5 s in the future; ref_time must be in the robot's clock.
        start_time = time.time() + 0.5
        ref_time = self._robot.time_sync.robot_timestamp_from_local_secs(start_time)
        robot_cmd = RobotCommandBuilder.arm_joint_move_helper(joint_positions=positions,
                                                              joint_velocities=velocities,
                                                              times=times,
                                                              ref_time=ref_time,
                                                              max_acc=MAX_JOINT_ACC,
                                                              max_vel=MAX_JOINT_VEL)
        cmd_id = self._command_client.robot_command(robot_cmd)
        block_until_arm_arrives(self._command_client, cmd_id, timeout_sec=T + 5.0)


def main() -> bool:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bosdyn.client.util.add_base_arguments(parser)
    opts = parser.parse_args()

    bosdyn.client.util.setup_logging(opts.verbose)
    sdk = create_standard_sdk("SpotThrows")
    robot = sdk.create_robot(opts.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert robot.has_arm(), "This script requires a Spot with an arm."

    with SpotThrows(robot, ) as ctl:
        ctl.power_on()
        ctl.stand()
        print("Pre-positioning arm to trajectory start...")
        ctl.initial_move()
        ctl.throw(T=1)
        time.sleep(5.0)
    return True


if __name__ == "__main__":
    try:
        ok = main()
    except (ResponseError, RpcError) as err:
        print(f"Robot communication error: {err}")
        ok = False
    sys.exit(0 if ok else 1)