"""OptiTrack-only check: connect to Motive and print what it streams. No robot.

  python mocap_check.py --local-ip 10.0.0.20 [--motive-ip 10.0.0.112] [--duration 10]

Prints the frame rate, rigid bodies and unlabeled markers (ball candidates)
once per second, then a summary.
"""

import argparse
import time

import numpy as np

from twin.natnet import NatNetClient


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local-ip", required=True)
    p.add_argument("--motive-ip", default="10.0.0.112")
    p.add_argument("--multicast", action="store_true", help="Motive set to multicast (lab uses unicast)")
    p.add_argument("--up-axis", choices=["y", "z"], default="z", help="Motive streaming up axis")
    p.add_argument("--duration", type=float, default=10.0, help="seconds to watch")
    p.add_argument("--command-port", type=int, default=1510, help="Motive Command Port")
    p.add_argument("--data-port", type=int, default=1511, help="Motive Data Port")
    opts = p.parse_args()

    client = NatNetClient(opts.motive_ip, opts.local_ip, multicast=opts.multicast, up_axis=opts.up_axis,
                          command_port=opts.command_port, data_port=opts.data_port).start()
    print(f"OptiTrack: {client.server_name}, NatNet {client.version}")
    try:
        t0 = time.time()
        frames_with_ball = 0
        total = 0
        last_print = t0
        while time.time() - t0 < opts.duration:
            time.sleep(0.05)
            now = time.time()
            if now - last_print < 1.0:
                continue
            batch = client.frames_between(last_print, now)
            last_print = now
            total += len(batch)
            frames_with_ball += sum(1 for f in batch if f.unlabeled())
            f = client.latest()
            if f is None:
                print("  no frames yet")
                continue
            rbs = ", ".join(f"#{rb.id} {np.round(rb.pos, 3)}{'' if rb.valid else ' (invalid)'}"
                            for rb in f.rigid_bodies.values()) or "none"
            balls = [np.round(m, 3) for m in f.unlabeled()]
            print(f"  {len(batch):4d} Hz | frame {f.frame_number} | rigid bodies: {rbs} | "
                  f"unlabeled markers ({len(balls)}): {balls}")
        print(f"\n{total} frames in {opts.duration:.0f} s ({total / opts.duration:.0f} Hz); "
              f"ball candidate visible in {100 * frames_with_ball / max(total, 1):.0f}% of frames")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
