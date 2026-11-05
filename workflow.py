"""Dataset loop: set the throw parameters -> throw forward on Spot -> the
landing distance (x) and the parameters go into a dataset.

  conda activate spotthrow
  python workflow.py

Spot is connected once at the start: it powers on, stands up and stays
standing between throws (arm stowed), so you can throw again and again.
q (or Ctrl+C) makes it sit down, power off and release the lease.

The throw is always the same straight-forward swing (twin/param_throw.py);
you change its timing / speeds / gripper open time before each throw.
Dataset: twin_data/dataset/throws.csv (one row per throw).

  python workflow.py --fake    offline test with a simulated Spot + OptiTrack
(Old flow, choosing a landing point: python workflow_target.py)
"""

import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "twin_data")
DSDIR = os.path.join(DATA, "dataset")
DATASET = os.path.join(DSDIR, "throws.csv")
SETTINGS = os.path.join(DSDIR, "current_params.json")
THROW_FILE = os.path.join(DSDIR, "current_throw.json")
PRED_DELAY = 0.25  # s, gripper delay used only for the simulated prediction ("c")

from twin.param_throw import ParamThrow  # noqa: E402

COLUMNS = ["id"] + ParamThrow.names() + ["landing_x"]  # throw number 1, 2, 3..., the 7 parameters, the result


def load_params():
    try:
        with open(SETTINGS) as f:
            return ParamThrow(**{k: v for k, v in json.load(f).items() if k in ParamThrow.names()})
    except (OSError, ValueError, TypeError):
        return ParamThrow()


def save_params(pt):
    os.makedirs(DSDIR, exist_ok=True)
    with open(SETTINGS, "w") as f:
        json.dump(pt.as_dict(), f, indent=2)


def rows():
    try:
        with open(DATASET, newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def append_row(row):
    os.makedirs(DSDIR, exist_ok=True)
    row = {**row, "id": len(rows()) + 1}
    new = not os.path.exists(DATASET)
    with open(DATASET, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def show_params(pt):
    T, rf, _ = pt.build()
    print(f"  throw parameters (motion {T:.2f} s, gripper open command at {rf * T:.2f} s):")
    for i, k in enumerate(ParamThrow.names(), 1):
        print(f"   {i}  {k:11s} = {getattr(pt, k):7.3f}   {ParamThrow.HELP[k]}")
    probs = pt.problems()
    print("  limits: " + ("OK" if not probs else "NOT OK -> " + "; ".join(probs)))


def edit(pt, idx):
    name = ParamThrow.names()[idx]
    s = input(f"{name} [{getattr(pt, name)}]: ").strip()
    if not s:
        return
    try:
        setattr(pt, name, float(s))
    except ValueError:
        print("Not a number, unchanged.")
        return
    save_params(pt)


def ask_float(prompt):
    while True:
        s = input(prompt).strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            print("Type a number in m, or just Enter.")


def throw(pt, lab):
    """One throw in the open session; Spot stays standing afterwards."""
    import run_twin as R
    probs = pt.problems()
    if probs:
        print("Not thrown, limits not OK: " + "; ".join(probs))
        return
    T, rf, traj = pt.build()
    opts = lab["opts"]
    opts.duration, opts.release_fraction, opts.plan_gripper_delay = T, rf, PRED_DELAY
    os.makedirs(DSDIR, exist_ok=True)
    with open(THROW_FILE, "w") as f:  # what was thrown last (same format as plan files)
        json.dump(pt.to_plan(PRED_DELAY), f, indent=2)
    if lab["ctl"] is not None:
        if not lab["ctl"]._robot.is_powered_on():
            print("Spot lost motor power (e-stop or power cut from outside this program). Not throwing.")
            print(robot_status(lab["ctl"]))
            return
        R.prepare_real_throw(lab["ctl"], traj, T, "[throw]")
    rec = lab["live"].throw(T, rf, trajectory=traj)
    if lab["ctl"] is not None:
        time.sleep(1.0)
        lab["ctl"].stow_arm()  # back to the stowed arm; Spot keeps standing
    try:
        R.save_record(rec, "live" if lab["ctl"] is not None else "fakelive", opts, 0)
    except Exception as err:  # the release analysis can fail when the ball is hidden; the landing still counts
        print(f"(release analysis skipped: {err})")
    from twin import ballistics
    ml = ballistics.measured_landing(rec.ball_t, rec.ball_pos, rec.floor_z + rec.params["ball_radius"])
    row = pt.as_dict()
    if ml is not None:
        row["landing_x"] = round(float(ml[1][0]), 4)
        print(f"\nLanding (OptiTrack): x = {row['landing_x']:.3f} m from Spot's body centre")
    else:
        print("\nOptiTrack did not see the landing.")
        x = ask_float("Measure it: type the landing x in metres (from Spot's body centre), or Enter to skip: ")
        if x is None:
            print("Skipped: not added to the dataset.")
            return
        row["landing_x"] = x
    append_row(row)
    print(f"Added throw #{len(rows())} to {os.path.relpath(DATASET, HERE)}")


def robot_status(ctl):
    """Why the robot stopped: motor power, e-stops, faults (read only)."""
    try:
        from bosdyn.api import robot_state_pb2
        from bosdyn.client.robot_state import RobotStateClient
        st = ctl._robot.ensure_client(RobotStateClient.default_service_name).get_robot_state()
        power = robot_state_pb2.PowerState.MotorPowerState.Name(st.power_state.motor_power_state)
        lines = [f"Spot status: motor power {power}"]
        for e in st.estop_states:
            lines.append(f"  e-stop {e.name}: {e.State.Name(e.state)}")
        for f in st.system_fault_state.faults:
            lines.append(f"  fault: {f.name} {f.error_message}")
        for f in st.behavior_fault_state.faults:
            lines.append(f"  behavior fault: {f.cause} {f.status}")
        if power != "STATE_ON":
            lines.append("  Spot is NOT powered: press q, fix the cause above, then start again.")
        return "\n".join(lines)
    except Exception as err:
        return f"(could not read Spot's status: {err})"


def lab_session(fake):
    """Context manager -> (robot interface, NatNet client, T_rb_body, rb id, ctl)."""
    import run_twin as R
    opts = SimpleNamespace(**R.LAB, multicast=False, external_estop=True, calib=str(R.CALIB_FILE),
                           max_vel=2.5, max_acc=10.0, ball="tennis", n=1, target=None)
    if fake:
        return opts, R.fake_lab(opts, ParamThrow().build()[2][1][0])
    return opts, R.real_lab(opts)


def menu(pt, lab):
    while True:
        print("\n" + "=" * 70)
        print(f"  dataset: {os.path.relpath(DATASET, HERE)} ({len(rows())} throws)   Spot: standing")
        show_params(pt)
        print("""
  1-7  change a parameter (kept for the next throws)
  t    THROW (Spot stays standing afterwards)   ** MOVES THE ARM **
  r    reset the parameters to the defaults
  q    quit: Spot sits down, powers off and releases the lease""")
        c = input("> ").strip().lower()
        if c.isdigit() and 1 <= int(c) <= len(ParamThrow.names()):
            edit(pt, int(c) - 1)
        elif c == "t":
            try:
                throw(pt, lab)
            except Exception as err:  # Spot keeps standing; q sits it down
                print(f"Throw failed: {err}")
                if lab["ctl"] is not None:
                    print(robot_status(lab["ctl"]))
        elif c == "r":
            pt = ParamThrow()
            save_params(pt)
        elif c == "q":
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="simulated Spot + OptiTrack (no robot)")
    fake = ap.parse_args().fake
    sys.path.insert(0, HERE)
    from twin.live import LiveSource
    pt = load_params()
    opts, session = lab_session(fake)
    print("Connecting to " + ("the simulated lab" if fake else f"Spot {opts.hostname}: it will power on and STAND UP"))
    try:
        with session as (robot, client, T, rb_id, ctl):
            live = LiveSource(robot, client, T, rb_id, ball=opts.ball)
            live.raw_path = os.path.join(DATA, "last_live_raw.npz")
            menu(pt, {"opts": opts, "live": live, "ctl": ctl})
            print("Sitting down...")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    print("Done. Spot sat down and is free for other programs." if not fake else "Done.")


if __name__ == "__main__":
    main()
