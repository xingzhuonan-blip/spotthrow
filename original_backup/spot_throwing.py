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


def elbow_traj(t:float, T:float=1):
    a0 = np.pi 
    a1 = 0 
    a2 = -1.5*np.pi/(T**2)
    a3 = np.pi/(T**3)
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
    joint_positions = [0, 0, 0, 0, 0, 0]
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
        desired = [0, -2.5, np.pi, 0, np.pi/2, 0]
        cmd = _make_arm_joint_cmd(desired, 3.0)
        cmd_id = self._command_client.robot_command(cmd)
        block_until_arm_arrives(self._command_client, cmd_id)
        
    
    def throw(self, T=1):
        start_time = time.time()
        dt=0.01
        N = 100
        gripper_is_open = False
        while time.time() - start_time < T:
            # Compute all the positions, velocities, and times we need for this segment of
            # the trajectory
            times = []
            positions = []
            velocities = []
            for i in range(N):
                t = segment_start_time + i * dt
                pos, vel = query_arm_joint_trajectory(t)
                positions.append(pos)
                velocities.append(vel)
                times.append(t)

            robot_cmd = RobotCommandBuilder.arm_joint_move_helper(joint_positions=positions,
                                                                  joint_velocities=velocities,
                                                                  times=times,
                                                                  max_acc=10000, max_vel=10000)

            # Wait until a bit before the previous trajectory is going to expire,
            # and send our new trajectory
            sleep_time = start_time + segment_start_time - (dt * 3 / 4) - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Send the request
            cmd_id = self._command_client.robot_command(robot_cmd)
            # We will start our next segment at the same place as the last point in our
            # trajectory, so we get good continuity
            segment_start_time = segment_start_time + dt * (N - 1)


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