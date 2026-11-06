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
from typing import Optional, Tuple

import numpy as np

import bosdyn.client.util
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient

from throw_trajectory import check_trajectory

ARM_JOINT_NAMES = ("arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1")

# Spot arm joint limits (radians), shrunk by a safety margin so we never
# command a point at the hard stop.
JOINT_LIMIT_MARGIN = 0.05
JOINT_LIMITS = {
    0: (-2.618, 3.141),   # sh0
    1: (-3.141, 0.524),   # sh1
    2: (0.0, 3.141),      # el0
    3: (-2.793, 2.793),   # el1
    4: (-1.833, 1.833),   # wr0
    5: (-2.880, 2.880),   # wr1
}

# Start pose: arm raised, elbow flexed (slightly inside the el0 limit,
# NOT exactly at it like the original script).
START_POSE = [0.0, -2.2, 3.0, 0.0, 0, 1.57]
ELBOW_START = START_POSE[2]
ELBOW_END = np.pi / 2  # swing of ~76 degrees

# Absolute caps we will never exceed, regardless of CLI flags.
from throw_trajectory import MAX_JOINT_VEL as HARD_MAX_JOINT_VEL, MAX_JOINT_ACC as HARD_MAX_JOINT_ACC


def elbow_traj(t: float, T: float) -> Tuple[float, float]:
    """Cubic from ELBOW_START to ELBOW_END with zero start/end velocity.

    q(t) = q0 + 3*d*(t/T)^2 - 2*d*(t/T)^3,  d = qf - q0
    Peak velocity = 1.5*|d|/T at t = T/2.
    """
    t = min(max(t, 0.0), T)
    d = ELBOW_END - ELBOW_START
    s = t / T
    angle = ELBOW_START + d * (3 * s**2 - 2 * s**3)
    vel = d * (6 * s - 6 * s**2) / T
    return angle, vel


def elbow_peak_acc(T: float) -> float:
    """Peak |acceleration| of the cubic, reached at t = 0 and t = T: 6*|d|/T^2."""
    return 6 * abs(ELBOW_END - ELBOW_START) / T**2


def query_arm_joint_trajectory(t: float, T: float):
    """Joint positions/velocities at time t. Only the elbow (el0) moves."""
    theta, omega = elbow_traj(t, T)
    joint_positions = list(START_POSE)
    joint_velocities = [0.0] * 6
    joint_positions[2] = theta
    joint_velocities[2] = omega
    return joint_positions, joint_velocities


def sample_trajectory(T: float, max_points: int = 200):
    """Sample the whole trajectory once (used both to validate and to send)."""
    n = min(max_points, max(20, int(T * 100)))  # ~50 Hz, capped
    times = list(np.linspace(0.0, T, n))
    positions, velocities = [], []
    for t in times:
        p, v = query_arm_joint_trajectory(t, T)
        positions.append(p)
        velocities.append(v)
    return times, positions, velocities


def validate_trajectory(times, positions, velocities, max_vel: float, max_acc: float) -> None:
    """Refuse to run if any sample violates joint limits, the velocity cap or the
    acceleration cap. If the SDK had to clamp acceleration, the executed motion
    (and therefore the release timing) would no longer match the plan."""
    T = times[-1]
    if elbow_peak_acc(T) > max_acc:
        raise ValueError(
            f"Peak elbow acceleration {elbow_peak_acc(T):.1f} rad/s^2 exceeds cap "
            f"{max_acc:.1f} rad/s^2. Increase --duration or raise --max-acc deliberately."
        )
    for k, (p, v) in enumerate(zip(positions, velocities)):
        for j in range(6):
            lo, hi = JOINT_LIMITS[j]
            if not (lo + JOINT_LIMIT_MARGIN <= p[j] <= hi - JOINT_LIMIT_MARGIN):
                raise ValueError(
                    f"Trajectory point {k}: joint {ARM_JOINT_NAMES[j]} = {p[j]:.3f} rad "
                    f"is outside safe range [{lo + JOINT_LIMIT_MARGIN:.3f}, "
                    f"{hi - JOINT_LIMIT_MARGIN:.3f}]."
                )
            if abs(v[j]) > max_vel:
                raise ValueError(
                    f"Trajectory point {k}: joint {ARM_JOINT_NAMES[j]} velocity "
                    f"{v[j]:.2f} rad/s exceeds cap {max_vel:.2f} rad/s. "
                    f"Increase --duration or raise --max-vel deliberately."
                )
    print(f"Trajectory validated: {len(times)} points, "
          f"peak elbow vel {max(abs(v[2]) for v in velocities):.2f} rad/s, "
          f"peak elbow acc {elbow_peak_acc(T):.2f} rad/s^2.")


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

    def initial_move(self, start_pose=START_POSE):
        """Move slowly to the start pose with conservative limits."""
        cmd = RobotCommandBuilder.arm_joint_move_helper(
            joint_positions=[list(start_pose)],
            times=[4.0],
            max_vel=2.0,
            max_acc=4.0,
        )
        cmd_id = self._command_client.robot_command(cmd)
        ok = block_until_arm_arrives(self._command_client, cmd_id, timeout_sec=10.0)
        if not ok:
            raise RuntimeError("Arm did not reach the start pose in time.")

    # Release opening: as fast as the gripper allows (the robot caps these at
    # its own limits). The default open is slow (~0.3 s from command to ball
    # free), and the ball rattles in the half-open gripper meanwhile.
    FAST_OPEN_VEL = 10.0   # rad/s
    FAST_OPEN_ACC = 100.0  # rad/s^2

    def open_gripper(self, fast=False):
        if fast:
            try:
                self._command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command(
                    max_vel=self.FAST_OPEN_VEL, max_acc=self.FAST_OPEN_ACC))
                return
            except (ResponseError, RpcError) as err:  # never keep the ball because of this
                print(f"  (fast open refused: {err}; opening normally)")
        self._command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())

    def close_gripper(self):
        self._command_client.robot_command(RobotCommandBuilder.claw_gripper_close_command())

    def throw(self, T: float, max_vel: float, max_acc: float,
              release: bool = False, release_fraction: float = 0.5, trajectory=None):
        """Execute the validated swing as a single joint trajectory.

        The whole trajectory is sampled and validated up front; the SDK is
        given finite max_vel/max_acc so it can clamp anything unexpected.

        trajectory: optional (times, positions, velocities), e.g. from
        throw_trajectory.cubic_joint_trajectory; default is the elbow swing.
        Returns local times of the swing start and of the gripper command.
        """
        # No ref_time: the robot starts the trajectory when it receives the
        # command. (Sending a start time converted to the robot clock put the
        # start ~263 s in the future on our Spot: the arm never moved while
        # the gripper opened on the PC clock.) start_time is set at send time
        # below; network latency (a few ms) is the timing error.
        if trajectory is None:
            times, positions, velocities = sample_trajectory(T)
            validate_trajectory(times, positions, velocities, max_vel, max_acc)
        else:
            times, positions, velocities = (np.asarray(x, float).tolist() for x in trajectory)
            problems = check_trajectory(times, positions, velocities, max_vel, max_acc)
            if problems:
                raise ValueError("Trajectory rejected: " + "; ".join(problems))
        release_cmd_time = None
        # Without ref_time the robot measures times from when it receives the
        # command and rejects a point at t = 0. That point is the start pose,
        # where the arm already is (initial_move), so drop it.
        times, positions, velocities = (list(x) for x in (times, positions, velocities))
        while times and times[0] <= 0.0:
            times, positions, velocities = times[1:], positions[1:], velocities[1:]

        robot_cmd = RobotCommandBuilder.arm_joint_move_helper(
            joint_positions=positions,
            joint_velocities=velocities,
            times=times,
            max_vel=max_vel,
            max_acc=max_acc,
        )

        start_time = time.time()
        cmd_id = self._command_client.robot_command(robot_cmd)

        if release:
            # Release is measured from the trajectory start (start_time), not
            # from when the command was sent.
            # NOTE: this is a separate command sent from the PC, so network
            # latency and gripper delay add to the actual release time. Verify on
            # the robot that it does not interrupt the arm trajectory.
            release_t = release_fraction * T
            sleep_for = start_time + release_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            release_cmd_time = time.time()
            self.open_gripper(fast=True)
            print(f"Gripper opened at t = {release_t:.2f} s.")

        arrived = block_until_arm_arrives(self._command_client, cmd_id, timeout_sec=T + 5.0)
        if not arrived:
            fb = self._command_client.robot_command_feedback(cmd_id)
            arm_fb = fb.feedback.synchronized_feedback.arm_command_feedback
            print("Robot command feedback:\n" + str(fb.feedback))
            raise RuntimeError(f"The arm did not execute the swing (status {fb.feedback.full_body_feedback} / "
                               f"{arm_fb}); see feedback above.")
        print("Swing complete.")
        return {"start_time": start_time, "release_cmd_time": release_cmd_time}


def main() -> bool:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Swing duration in seconds (default 3.0; start SLOW).")
    parser.add_argument("--max-vel", type=float, default=8.0,
                        help="Joint velocity limit passed to the SDK (rad/s).")
    parser.add_argument("--max-acc", type=float, default=10.0,
                        help="Joint acceleration limit passed to the SDK (rad/s^2).")
    parser.add_argument("--release", action="store_true",
                        help="Open the gripper mid-swing to actually throw.")
    parser.add_argument("--release-fraction", type=float, default=0.5,
                        help="When to open the gripper, as a fraction of the swing (0-1).")
    parser.add_argument("--external-estop", action="store_true",
                        help="Do NOT take a software e-stop; assumes a tablet or "
                             "physical e-stop is already registered (recommended).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print the trajectory without contacting the robot.")
    parser.add_argument("--plan",
                        help="Throw plan from 'run_twin.py plan' (twin_data/plans/*.json): sets the "
                             "trajectory, --duration and --release-fraction.")
    opts = parser.parse_args()

    # Clamp user-provided limits to hard caps; never allow 'unlimited'.
    max_vel = min(opts.max_vel, HARD_MAX_JOINT_VEL)
    max_acc = min(opts.max_acc, HARD_MAX_JOINT_ACC)

    trajectory = None
    start_pose = START_POSE
    if opts.plan:
        import json
        from throw_trajectory import cubic_joint_trajectory
        plan = json.loads(open(opts.plan).read())
        opts.duration, opts.release_fraction = plan["duration"], plan["release_fraction"]
        trajectory = cubic_joint_trajectory(plan["start_pose"], plan["end_pose"], opts.duration)
        start_pose = plan["start_pose"]
        print(f"Plan {opts.plan}: target {plan['target']}, T = {opts.duration:.2f} s, "
              f"release at {opts.release_fraction:.2f} of the swing")
        print(f"  start joints {[round(v, 3) for v in plan['start_pose']]}")
        print(f"  end joints   {[round(v, 3) for v in plan['end_pose']]}")

    if opts.duration < 0.4:
        print("Refusing durations below 0.4 s.")
        return False
    if not (0.0 < opts.release_fraction < 1.0):
        print("--release-fraction must be between 0 and 1.")
        return False

    # Validate the trajectory before touching the robot at all.
    if trajectory is None:
        times, positions, velocities = sample_trajectory(opts.duration)
        try:
            validate_trajectory(times, positions, velocities, max_vel, max_acc)
        except ValueError as err:
            print(f"Trajectory rejected: {err}")
            return False
    else:
        problems = check_trajectory(*trajectory, max_vel, max_acc)
        if problems:
            print("Trajectory rejected: " + "; ".join(problems))
            return False
        peak = max(abs(v) for row in trajectory[2] for v in row)
        print(f"Trajectory validated: {len(trajectory[0])} points, peak joint speed {peak:.2f} rad/s.")

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
                ctl.initial_move(start_pose)
                input(f"Ready to swing in {opts.duration:.1f} s. CLEAR THE AREA, "
                    f"then press Enter (Ctrl+C aborts safely)...")
                ctl.throw(T=opts.duration, max_vel=max_vel, max_acc=max_acc,
                        release=opts.release, release_fraction=opts.release_fraction,
                        trajectory=trajectory)
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
