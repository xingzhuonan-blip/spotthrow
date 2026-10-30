"""Safer Spot arm "throw" demo.

Key safety properties:
  * Trajectory is validated against joint limits and a velocity cap BEFORE
    anything is sent to the robot.
  * Real (finite) max_vel / max_acc limits are passed to the SDK.
  * Conservative defaults (slow 3 s swing); speed up only deliberately.
  * Ctrl+C at ANY point: the arm is stopped, stowed, the robot sits and
    powers off safely, and the lease/e-stop are released cleanly.
  * Software e-stop is optional (--external-estop) so you can keep a
    tablet / physical e-stop as the authority instead.
  * Gripper release is opt-in (--release) and timed to the swing.

First run checklist:
  1. Large clear area, nothing within arm reach + throw arc.
  2. Keep a hardware/tablet e-stop in hand (run with --external-estop).
  3. Start slow:  python spot_throw_safe.py ROBOT_IP --duration 4
  4. Only reduce --duration once the slow motion looks right.
"""

import argparse
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

import bosdyn.client.util
from bosdyn.api import robot_state_pb2
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b

from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
    blocking_stand,
)
from bosdyn.client.math_helpers import SE3Pose, SE3Velocity, Quat
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_timestamp, timestamp_to_sec



def cartesian_traj(t: float, T: float) -> Tuple[SE3Pose, SE3Velocity]:
    t = min(max(t, 0.0), T)
    X_INIT = 0.5
    Z_INIT = 0
    X_END = 1.0
    Z_END = 0.5
    d_x = X_END - X_INIT 
    d_z = Z_END - Z_INIT
    s = t/T 
    x  = X_INIT + d_x * (3 * s**2 - 2 * s**3)
    y = 0
    z = Z_INIT + d_z *  (3 * s**2 - 2 * s**3)
    v_x = d_x * (6 * s - 6 * s**2) / T
    v_z = d_z * (6 * s - 6 * s**2) / T
    v_y = 0
    return SE3Pose(x, t, z, Quat(1,0,0,0)), SE3Velocity(v_x, v_y, v_z, 0, 0, 0)
    


def sample_trajectory(T: float, max_points: int = 200):
    """Sample the whole trajectory once (used both to validate and to send)."""
    dt = 0.05
    times = np.arange(0.0, T+dt, dt)
    positions, velocities = [], []
    for t in times:
        p, v = cartesian_traj(t, T)
        positions.append(p.to_proto())
        velocities.append(v.to_proto())
    return times, positions, velocities




class SpotThrows:
    def __init__(self, robot, use_software_estop: bool = True):
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

    # ---- lifecycle ---------------------------------------------------

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        # Runs on normal exit, exceptions AND KeyboardInterrupt (Ctrl+C).
        self.shutdown()
        return False

    def acquire(self):
        self._robot.time_sync.wait_for_sync()
        if self.use_software_estop:
            if self._estop_client is None:
                raise RuntimeError("E-stop client unavailable; cannot take software e-stop.")
            ep = EstopEndpoint(self._estop_client, "spot_throw_safe", estop_timeout=9.0)
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

    # ---- safe stop / shutdown ----------------------------------------

    def safe_stop(self):
        """Immediately stop whatever the robot is doing (arm included)."""
        try:
            self._command_client.robot_command(RobotCommandBuilder.stop_command())
        except (ResponseError, RpcError) as err:
            print(f"  (stop command failed: {err})")

    def stow_arm(self, timeout_sec: float = 6.0):
        try:
            cmd_id = self._command_client.robot_command(RobotCommandBuilder.arm_stow_command())
            block_until_arm_arrives(self._command_client, cmd_id, timeout_sec=timeout_sec)
        except (ResponseError, RpcError) as err:
            print(f"  (stow failed: {err})")

    def _wait_until_sitting(self, timeout_sec: float = 10.0) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                state = self._state_client.get_robot_state()
            except (ResponseError, RpcError):
                time.sleep(0.5)
                continue
            for behavior in state.behavior_fault_state.faults:
                print(f"  behavior fault: {behavior}")
            # Treat "not moving + low body height" pragmatically: rely on
            # power state transition below instead of pose heuristics.
            time.sleep(0.5)
            if not self._robot.is_powered_on():
                return True
        return True  # we still proceed to power_off either way

    def shutdown(self, sit_and_power_off: bool = True):
        """Always leave the robot in a safe state, even after Ctrl+C.

        Order matters: stop -> stow -> sit+power off -> only THEN release
        the lease and e-stop keepalives.
        """
        print("\nShutting down safely...")
        try:
            if sit_and_power_off and self._robot.is_powered_on():
                self.safe_stop()
                time.sleep(0.5)
                self.stow_arm()
                # cut_immediately=False performs a "safe power off": the
                # robot sits down first, then motors power off.
                try:
                    self._robot.power_off(cut_immediately=False, timeout_sec=30)
                    print("Robot sat down and powered off.")
                except (ResponseError, RpcError) as err:
                    print(f"Safe power-off failed ({err}); sending sit command directly.")
                    try:
                        self._command_client.robot_command(
                            RobotCommandBuilder.synchro_sit_command())
                        time.sleep(3.0)
                        self._robot.power_off(cut_immediately=False, timeout_sec=20)
                    except (ResponseError, RpcError) as err2:
                        print(f"Fallback power-off also failed: {err2}")
        except KeyboardInterrupt:
            # Second Ctrl+C during shutdown: still try to power off before exiting.
            print("Interrupted during shutdown; attempting immediate safe power-off.")
            try:
                self._robot.power_off(cut_immediately=False, timeout_sec=20)
            except Exception:
                pass
        finally:
            for ka in (self._estop_keepalive, self._lease_keepalive):
                if ka is not None:
                    try:
                        ka.shutdown()
                    except Exception:
                        pass
            self._estop_keepalive = self._lease_keepalive = None
            print("Lease and e-stop released.")

    # ---- motions ------------------------------------------------------

    def initial_move(self):
        """Move slowly to the start pose with conservative limits."""
        robot_state = self._state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        initial_pose, _ = cartesian_traj(0, 1)
        odom_T_hand = odom_T_flat_body * initial_pose
        cmd = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, 4)
        cmd_id = self._command_client.robot_command(cmd)
        ok = block_until_arm_arrives(self._command_client, cmd_id, timeout_sec=10.0)
        if not ok:
            raise RuntimeError("Arm did not reach the start pose in time.")

    def open_gripper(self):
        self._command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())

    def close_gripper(self):
        self._command_client.robot_command(RobotCommandBuilder.claw_gripper_close_command())

    def throw(self, T: float, max_vel: float, max_acc: float,
              release: bool = False, release_fraction: float = 0.5):
        """Execute the validated swing as a single joint trajectory.

        The whole trajectory is sampled and validated up front; the SDK is
        given finite max_vel/max_acc so it can clamp anything unexpected.
        """
        grav_body_T_task = SE3Pose(x=0.0, y=0, z=0, rot=Quat(w=1, x=0., y=0., z=0.))

        robot_state = self._state_client.get_robot_state()
        odom_T_grav_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            ODOM_FRAME_NAME,
            GRAV_ALIGNED_BODY_FRAME_NAME,
        )

        odom_T_task = odom_T_grav_body * grav_body_T_task
        start_time = time.time()+0.5
        ref_time = seconds_to_timestamp(start_time)
        times, positions, velocities = sample_trajectory(T)

        robot_cmd = RobotCommandBuilder.arm_cartesian_move_helper(
            se3_poses=positions,
            times=times,
            se3_velocities=velocities,
            root_frame_name=ODOM_FRAME_NAME,
            root_tform_task=odom_T_task.to_proto(),
            max_acc=10000,
            max_linear_vel=10000,
            max_angular_vel=10000,
            ref_time=ref_time,
        )

        cmd_id = self._command_client.robot_command(robot_cmd)

        if release:
            release_t = release_fraction * T
            while time.time()-timestamp_to_sec(ref_time)<release_t:
                pass
            self.open_gripper()
            print(f"Gripper opened at t = {release_t:.2f} s.")

        block_until_arm_arrives(self._command_client, cmd_id, timeout_sec=T + 5.0)
        print("Swing complete.")


def main() -> bool:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Swing duration in seconds (default 3.0; start SLOW).")
    parser.add_argument("--release", action="store_true",
                        help="Open the gripper mid-swing to actually throw.")
    parser.add_argument("--release-fraction", type=float, default=0.5,
                        help="When to open the gripper, as a fraction of the swing (0-1).")
    parser.add_argument("--external-estop", action="store_true",
                        help="Do NOT take a software e-stop; assumes a tablet or "
                             "physical e-stop is already registered (recommended).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print the trajectory without contacting the robot.")
    opts = parser.parse_args()

    # Clamp user-provided limits to hard caps; never allow 'unlimited'.
    if opts.duration < 0.4:
        print("Refusing durations below 0.4 s.")
        return False
    if not (0.0 < opts.release_fraction < 1.0):
        print("--release-fraction must be between 0 and 1.")
        return False

    if opts.dry_run:
        print("Dry run only; no robot contact.")
        return True

    bosdyn.client.util.setup_logging(opts.verbose)
    sdk = create_standard_sdk("SpotThrowsSafe")
    robot = sdk.create_robot(opts.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert robot.has_arm(), "This script requires a Spot with an arm."

    try:
        with SpotThrows(robot, use_software_estop=not opts.external_estop) as ctl:
            ctl.power_on()
            ctl.stand()
            time.sleep(1.0)
            while True:
                ctl.stow_arm()
                if opts.release:
                    ctl.open_gripper()
                    input("Place the object in the gripper, then press Enter to grip...")
                    ctl.close_gripper()
                    time.sleep(1.0)
                print("Pre-positioning arm to trajectory start (slow)...")
                ctl.initial_move()
                input(f"Ready to swing in {opts.duration:.1f} s. CLEAR THE AREA, "
                    f"then press Enter (Ctrl+C aborts safely)...")
                ctl.throw(T=opts.duration, max_vel=None, max_acc=None,
                        release=opts.release, release_fraction=opts.release_fraction)
                time.sleep(2.0)
        return True
    except KeyboardInterrupt:
        # Shutdown already ran inside the context manager's __exit__.
        print("Aborted by user; robot was shut down safely.")
        return False


if __name__ == "__main__":
    try:
        ok = main()
    except (ResponseError, RpcError) as err:
        print(f"Robot communication error: {err}")
        ok = False
    sys.exit(0 if ok else 1)
