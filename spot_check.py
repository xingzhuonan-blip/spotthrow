"""Read-only Spot check: log in and print the robot's state. Never moves it.

  python spot_check.py [10.0.0.30]

No lease, no power on, no commands. Credentials come from
BOSDYN_CLIENT_USERNAME / BOSDYN_CLIENT_PASSWORD, or you are prompted.
"""

import sys

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient


def main():
    hostname = sys.argv[1] if len(sys.argv) > 1 else "10.0.0.30"
    sdk = bosdyn.client.create_standard_sdk("SpotReadOnlyCheck")
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    print(f"Logged in to {hostname}")

    print("ID:", robot.get_id().nickname or "", "| serial", robot.get_id().serial_number)
    print("has arm:", robot.has_arm())
    print("motors powered:", robot.is_powered_on())
    print("E-stopped:", robot.is_estopped())

    state = robot.ensure_client(RobotStateClient.default_service_name).get_robot_state()
    bat = state.battery_states[0] if state.battery_states else None
    if bat:
        print(f"battery: {bat.charge_percentage.value:.0f}%")
    print("power state:", state.power_state)
    faults = list(state.system_fault_state.faults)
    print(f"system faults: {len(faults)}")
    for f in faults:
        print("  -", f.name, f.error_message)
    print("arm joints:")
    for j in state.kinematic_state.joint_states:
        if j.name.startswith("arm0"):
            print(f"  {j.name:12s} {j.position.value:+.3f} rad")


if __name__ == "__main__":
    main()
