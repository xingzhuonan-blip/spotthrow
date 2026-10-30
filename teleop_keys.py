"""Slow keyboard driving for Spot. Run in a real terminal.

  python teleop_keys.py [10.0.0.30]

  Up / Down     walk forward / backward
  Left / Right  turn left / right
  a / d         step sideways left / right
  Space         stop
  o / c         open / close the gripper (to put the ball in)
  q             sit, power off and quit (Ctrl+C does the same)

Each key press walks for 0.6 s and then stops, so releasing the keys stops the
robot. Speeds are capped low. The tablet / hardware e-stop stays the authority.
"""

import curses
import sys
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand

V_LIN = 0.3   # m/s
V_ROT = 0.4   # rad/s
HOLD = 0.6    # s each key press keeps moving

KEYS = {
    curses.KEY_UP: (V_LIN, 0, 0, "forward"),
    curses.KEY_DOWN: (-V_LIN, 0, 0, "backward"),
    curses.KEY_LEFT: (0, 0, V_ROT, "turn left"),
    curses.KEY_RIGHT: (0, 0, -V_ROT, "turn right"),
    ord("a"): (0, V_LIN, 0, "step left"),
    ord("d"): (0, -V_LIN, 0, "step right"),
}


def drive(stdscr, robot, cmd):
    stdscr.nodelay(True)
    stdscr.addstr(0, 0, "Arrows: walk/turn | a/d: sideways | o/c: gripper | Space: stop | q: sit and quit")
    while True:
        key = stdscr.getch()
        if key == ord("q"):
            return
        if key in KEYS:
            vx, vy, vr, name = KEYS[key]
            cmd.robot_command(RobotCommandBuilder.synchro_velocity_command(v_x=vx, v_y=vy, v_rot=vr),
                              end_time_secs=time.time() + HOLD)
            status = name
        elif key == ord("o"):
            cmd.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0))
            status = "gripper open"
        elif key == ord("c"):
            cmd.robot_command(RobotCommandBuilder.claw_gripper_close_command())
            status = "gripper close"
        elif key == ord(" "):
            cmd.robot_command(RobotCommandBuilder.stop_command())
            status = "stop"
        else:
            time.sleep(0.03)
            continue
        stdscr.move(2, 0)
        stdscr.clrtoeol()
        stdscr.addstr(2, 0, f"> {status}")
        stdscr.refresh()


def main():
    hostname = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.30"
    sdk = bosdyn.client.create_standard_sdk("SpotTeleopKeys")
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    if robot.is_estopped():
        sys.exit("Robot is e-stopped; release the e-stop first.")

    cmd = robot.ensure_client(RobotCommandClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        try:
            print("Powering on and standing...")
            robot.power_on(timeout_sec=20)
            blocking_stand(cmd, timeout_sec=10)
            curses.wrapper(drive, robot, cmd)
        except KeyboardInterrupt:
            pass
        finally:
            print("Sitting and powering off...")
            robot.power_off(cut_immediately=False, timeout_sec=20)
            print("Powered off. Lease released.")


if __name__ == "__main__":
    main()
