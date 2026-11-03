"""One terminal for the whole loop: plan (+ simulation preview) -> throw -> compare.

  conda activate spotthrow
  python workflow.py

Each step runs run_twin.py and comes back to the menu when its window is
closed. Only step 2 moves Spot (it asks for Enter twice, as usual).
"""

import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(HERE, "twin_data", "workflow_settings.json")


def load_delay():
    try:
        with open(SETTINGS) as f:
            return str(json.load(f)["gripper_delay"])
    except (OSError, ValueError, KeyError):
        return "0.1"


def save_delay(delay):
    with open(SETTINGS, "w") as f:
        json.dump({"gripper_delay": float(delay)}, f)


def newest(pattern):
    files = glob.glob(os.path.join(HERE, pattern))
    return max(files, key=os.path.getmtime) if files else None


def run(*args):
    cmd = [sys.executable, os.path.join(HERE, "run_twin.py"), *args]
    print("\n$ " + " ".join(cmd[1:]) + "\n")
    try:
        subprocess.run(cmd, cwd=HERE)
    except KeyboardInterrupt:
        print("\n(stopped)")


def show(name, path):
    print(f"  {name:12s} {os.path.basename(path) if path else '-'}")


def main():
    delay = load_delay()  # remembered between runs (twin_data/workflow_settings.json)
    while True:
        print("\n" + "=" * 60)
        show("newest plan", newest("twin_data/plans/*.json"))
        show("newest live", newest("twin_data/live_*.npz"))
        print(f"""
  1  plan      click a target (preview = physics simulation), e = export   (gripper delay {delay} s)
              mouse: right drag rotate, wheel-button drag pan, wheel zoom
  2  throw     throw the newest plan on Spot + record   ** MOVES SPOT **
  3  compare   view the newest live throw (green = plan, orange = ball)
  d  change the gripper delay (saved as the default for every next plan)
  q  quit""")
        try:
            c = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if c == "1":
            run("plan", "--gripper-delay", delay)
        elif c == "2":
            run("throw", "--record")
        elif c == "3":
            live = newest("twin_data/live_*.npz")
            run("view", live) if live else print("No live throw yet.")
        elif c == "d":
            new = input(f"gripper delay in s [{delay}]: ").strip() or delay
            try:
                float(new)
            except ValueError:
                print("Not a number, kept " + delay)
                continue
            delay = new
            save_delay(delay)
            print(f"Saved {delay} s. Press 1 to make a new plan with it.")
        elif c == "q":
            return


if __name__ == "__main__":
    main()
