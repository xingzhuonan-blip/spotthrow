"""Stand Spot up and hold still until Ctrl+C, then sit and power off.

  python stand_hold.py [10.0.0.30]

No arm or leg motion beyond standing. The tablet / hardware e-stop stays the
authority (no software e-stop is registered).
"""

import sys
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand


def main():
    hostname = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.30"
    sdk = bosdyn.client.create_standard_sdk("SpotStandHold")
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    if robot.is_estopped():
        sys.exit("Robot is e-stopped; release the e-stop first.")

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    cmd = robot.ensure_client(RobotCommandClient.default_service_name)
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        try:
            print("Powering on...")
            robot.power_on(timeout_sec=20)
            print("Standing...")
            blocking_stand(cmd, timeout_sec=10)
            print("Standing. Holding still. Press Ctrl+C to sit and power off.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            print("Sitting and powering off...")
            robot.power_off(cut_immediately=False, timeout_sec=20)
            print("Powered off. Lease released.")


if __name__ == "__main__":
    main()
