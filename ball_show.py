"""Hold the ball up (gripper pointing up) and move it slowly in a loop, while
checking whether OptiTrack sees it. Spot does not walk.

  python ball_show.py 10.0.0.30 --local-ip 10.0.0.20 --data-port 9000 --body-rb 12

Steps: stand -> open gripper, you place the ball, Enter -> close -> raise the
arm slowly (4 s) -> 2 slow loops (12 points, 1.5 s each) -> stow, sit, power off.
Ctrl+C at any time stops, stows, sits and powers off. Tablet / hardware e-stop
stays the authority (no software e-stop).

Poses: checked in simulation for joint limits and self-collision; gripper
points straight up, about 1.1 m above the body, loop about 20 x 16 cm.
"""

import argparse
import time

import numpy as np

S1, E0, K = -1.35, 0.35, -1.62   # sh1, el0 centre; sh1 + el0 + wr0 = K keeps the gripper pointing up
UP = [0.0, S1, E0, 0.0, K - S1 - E0, 0.0]
LOOP = [[0.35 * np.sin(t), S1 + 0.15 * np.cos(t), E0, 0.0, K - (S1 + 0.15 * np.cos(t)) - E0, 0.0]
        for t in np.linspace(0, 2 * np.pi, 13)[:-1]]


def high_points(frames, body_xy=None, min_z=0.9):
    """Unlabeled points above min_z (m). Spot's body markers sit at about
    0.6 m, the raised ball at about 1.6 m, so height alone separates them."""
    return [[p for p in f.unlabeled() if p[2] > min_z] for f in frames]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hostname")
    ap.add_argument("--local-ip", required=True)
    ap.add_argument("--motive-ip", default="10.0.0.112")
    ap.add_argument("--data-port", type=int, default=1511)
    ap.add_argument("--body-rb", type=int, required=True)
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--step-s", type=float, default=1.5, help="seconds per loop point (slow)")
    opts = ap.parse_args()

    import bosdyn.client.util
    from bosdyn.client import create_standard_sdk

    import spot_throw_safe as real
    from twin.live import SpotRobotInterface, _rb_pose
    from twin.natnet import NatNetClient

    client = NatNetClient(opts.motive_ip, opts.local_ip, data_port=opts.data_port).start()
    print(f"OptiTrack: {client.server_name}, NatNet {client.version}")
    sdk = create_standard_sdk("SpotBallShow")
    robot = sdk.create_robot(opts.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    try:
        with real.SpotThrows(robot, use_software_estop=False) as ctl:
            arm = SpotRobotInterface(ctl)
            ctl.power_on()
            ctl.stand()
            ctl.open_gripper()
            input("Put the ball in the gripper, then press Enter to close it...")
            ctl.close_gripper()
            time.sleep(1.0)
            input("The arm will rise SLOWLY and loop above Spot. Clear the area, then press Enter...")

            t0 = time.time()
            time.sleep(0.5)
            T_rb = _rb_pose(client.frames_between(t0, time.time()), opts.body_rb)
            body_xy = T_rb[:3, 3][:2] if T_rb is not None else np.array([np.inf, np.inf])

            arm.move_arm(UP, seconds=4.0)
            time.sleep(1.0)
            t_start = time.time()
            rows = []
            for loop in range(opts.loops):
                for k, q in enumerate(LOOP):
                    ts = time.time()
                    arm.move_arm(q, seconds=opts.step_s)
                    pts = high_points(client.frames_between(ts, time.time()), body_xy)
                    seen = sum(1 for p in pts if p)
                    last = next((p[0] for p in reversed(pts) if p), None)
                    rows.append(seen / max(len(pts), 1))
                    print(f"  loop {loop + 1} point {k + 1:2d}: ball seen in {100 * rows[-1]:3.0f}% of "
                          f"{len(pts)} frames, last at {np.round(last, 3) if last is not None else '-'}")
            all_pts = high_points(client.frames_between(t_start, time.time()), body_xy)
            seen = [p[0] for p in all_pts if p]
            print(f"\nwhole loop: ball seen in {100 * len(seen) / max(len(all_pts), 1):.0f}% of "
                  f"{len(all_pts)} frames")
            if len(seen) > 10:
                P = np.array(seen)
                print(f"  ball path: x {P[:, 0].min():.3f}..{P[:, 0].max():.3f}, "
                      f"y {P[:, 1].min():.3f}..{P[:, 1].max():.3f}, z {P[:, 2].min():.3f}..{P[:, 2].max():.3f} m")
                np.save("ball_show_path.npy", P)
                print("  saved ball_show_path.npy")
            ctl.stow_arm()
    finally:
        client.stop()


if __name__ == "__main__":
    main()
