"""Spot throwing digital twin.

Offline (no hardware):
  python run_twin.py simulate --duration 1.0 --release-fraction 0.5   # simulated throw record
  python run_twin.py simulate --start-pose 0 -2.2 3.0 0 0 1.57 --end-pose 0 -1.2 1.8 0 0 1.57
  python run_twin.py fake-live --live-view                            # live pipeline vs a fake lab
  python run_twin.py optimize --source sim --target 0.95 0.15         # learn to hit a target
  python run_twin.py optimize --source fake-live --target 0.95 0.15 --live-view
  python run_twin.py plan                                             # click a target -> throw
  python run_twin.py simulate --plan twin_data/plans/plan_*.json      # run an exported plan
  python run_twin.py analyze twin_data/*.npz                          # print the analysis
  python run_twin.py view                                             # replay records in 3D

Short commands (lab settings in LAB below; the newest plan unless a file is given):
  python run_twin.py plan                    # click a target, press e to export
  python run_twin.py fast-plan --target 1.5 0 # or: a fast overhand throw to (x, y)
  python run_twin.py sim                     # simulate the newest plan + 3D replay
  python run_twin.py throw                   # throw it on Spot (no OptiTrack needed)
  python run_twin.py throw --record          # ...and record it with OptiTrack + 3D window

In the lab (PC on the SpotRoom network, Motive streaming unicast, ball =
single marker, Spot's body = rigid body; see twin/live.py):
  python run_twin.py calibrate 10.0.0.30 --local-ip <this PC> --body-rb <id>
  python run_twin.py live 10.0.0.30 --local-ip <this PC> --duration 2 --live-view
  python run_twin.py optimize --source live --hostname 10.0.0.30 --local-ip <this PC> --target 0.95 0 --live-view

Records go to twin_data/, all in Spot's robot frame (see twin/record.py).
"""

import argparse
import contextlib
import glob
import sys
import time
from pathlib import Path

import numpy as np

from twin import ballistics
from twin.record import ThrowRecord
from twin.sources import FakeOptiTrack, SimSource
from twin.viewer import Viewer, summary_lines

DATA_DIR = Path(__file__).resolve().parent / "twin_data"
CALIB_FILE = Path(__file__).resolve().parent / "calibration.json"
FORWARD_THROW = ([0, -2.2, 3.0, 0, 0, 1.57], [0, -1.2, 1.8, 0, 0, 1.57])  # found in simulation

# Lab settings used by the short commands 'sim' and 'throw' (edit here if the lab changes).
LAB = {"hostname": "10.0.0.30", "local_ip": "10.0.0.20", "motive_ip": "10.0.0.112", "data_port": 9000}


def trajectory_from(opts):
    """--plan file, or custom start/end pose -> cubic joint trajectory;
    None = spot_throw_safe default. A plan also sets duration, release and target."""
    if getattr(opts, "plan", None):
        from twin.planner import load_plan
        d, traj = load_plan(opts.plan)
        opts.duration, opts.release_fraction = d["duration"], d["release_fraction"]
        opts.plan_gripper_delay = d["gripper_delay"]
        if opts.target is None:
            opts.target = d["target"]
        tgt = "none" if opts.target is None else np.round(opts.target, 3)
        print(f"plan {opts.plan}: T={opts.duration:.2f} s, release {opts.release_fraction:.2f}, target {tgt}")
        return traj
    if opts.start_pose is None and opts.end_pose is None:
        return None
    if opts.start_pose is None or opts.end_pose is None:
        sys.exit("Give both --start-pose and --end-pose.")
    from throw_trajectory import cubic_joint_trajectory
    return cubic_joint_trajectory(opts.start_pose, opts.end_pose, opts.duration)


def save_record(rec, prefix, opts, i, folder=None):
    if getattr(opts, "plan_gripper_delay", None) is not None:
        rec.params["plan_gripper_delay"] = opts.plan_gripper_delay  # viewer: planned flight
    name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_T{opts.duration}_rf{opts.release_fraction}_{i}.npz"
    out = Path(folder) if folder else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = rec.save(out / name)
    opts.last_record = path
    print(f"saved {path}")
    return print_analysis(rec)


def print_analysis(rec):
    P = rec.params
    a = ballistics.analyze(rec, P["ball_radius"], P["ball_mass"], P["drag_coeff"])
    for line in summary_lines(rec, a):
        print("  " + line)
    return a


# ---- sessions: fake lab / real lab ---------------------------------------
# Each yields (robot interface, NatNet client, T_rb_body, rigid body id, ctl or None).

@contextlib.contextmanager
def fake_lab(opts, start_pose):
    from twin.fake_world import FakeMotive, FakeRobot, FakeWorld
    from twin.live import run_calibration
    from twin.natnet import NatNetClient

    world = FakeWorld(SimSource(ball=opts.ball), start_pose=start_pose, up_axis="y")
    motive = FakeMotive(world, command_port=15100).start()
    client = NatNetClient("127.0.0.1", "127.0.0.1", command_port=15100, data_port=15111, up_axis="y")
    try:
        client.start()
        print(f"connected to {client.server_name}, NatNet {client.version}")
        robot = FakeRobot(world)
        print("calibrating body markers...")
        T, rms, _ = run_calibration(robot, client, world.rb_id, settle_s=0.2, average_s=0.3, verbose=False)
        err = np.linalg.norm(T[:3, 3] - world.T_rb_body[:3, 3])
        print(f"calibration rms {rms * 1000:.1f} mm (offset error vs truth {err * 1000:.1f} mm)")
        yield robot, client, T, world.rb_id, None
    finally:
        client.stop()
        motive.stop()


def spot_session(opts):
    """Authenticated SpotThrows controller (use as a context manager)."""
    import bosdyn.client.util
    from bosdyn.client import create_standard_sdk

    import spot_throw_safe as real
    sdk = create_standard_sdk("SpotThrowTwin")
    robot = sdk.create_robot(opts.hostname)
    bosdyn.client.util.authenticate(robot)  # BOSDYN_CLIENT_USERNAME / _PASSWORD, or prompt
    robot.time_sync.wait_for_sync()
    assert robot.has_arm(), "This needs a Spot with an arm."
    return real.SpotThrows(robot, use_software_estop=not opts.external_estop)


def mocap_client(opts, up_axis):
    from twin.natnet import NatNetClient
    client = NatNetClient(opts.motive_ip, opts.local_ip, multicast=opts.multicast, up_axis=up_axis,
                          data_port=opts.data_port).start()
    print(f"OptiTrack: {client.server_name}, NatNet {client.version}")
    return client


@contextlib.contextmanager
def real_lab(opts):
    from twin import calibration
    from twin.live import SpotRobotInterface

    if not opts.hostname or not opts.local_ip:
        sys.exit("The real lab needs the Spot address (hostname) and --local-ip.")
    T, rb_id, up_axis, rms = calibration.load(opts.calib)
    print(f"calibration {opts.calib}: rigid body {rb_id}, residual {rms * 1000:.1f} mm")
    client = mocap_client(opts, up_axis)
    try:
        with spot_session(opts) as ctl:
            ctl.power_on()
            ctl.stand()
            yield SpotRobotInterface(ctl, max_vel=opts.max_vel, max_acc=opts.max_acc), client, T, rb_id, ctl
    finally:
        client.stop()


def prepare_real_throw(ctl, traj, duration, label):
    """Operator steps before every real throw."""
    ctl.stow_arm()
    ctl.open_gripper()
    input(f"{label} Put the ball in the gripper, then press Enter to grip...")
    ctl.close_gripper()
    time.sleep(1.0)
    ctl.initial_move(traj[1][0])
    input(f"Ready to throw (T = {duration:.2f} s). CLEAR THE AREA, then press Enter (Ctrl+C aborts safely)...")


def with_optional_view(opts, robot, client, T, rb_id, target, work):
    """Run work(view_or_None) with or without the real-time window."""
    if not opts.live_view:
        return work(None)
    from twin.live_view import LiveView
    LiveView(robot, client, T, rb_id, target=target).run(work)


# ---- offline commands ------------------------------------------------------

def cmd_simulate(opts):
    src = SimSource(ball=opts.ball, gripper_delay=opts.gripper_delay,
                    optitrack=FakeOptiTrack(rate_hz=opts.rate, noise_m=opts.noise, dropout=opts.dropout))
    traj = trajectory_from(opts)  # first: a --plan sets duration / release / target
    for i in range(opts.n):
        rec = src.throw(opts.duration, opts.release_fraction, target=opts.target, trajectory=traj)
        save_record(rec, "sim", opts, i)


def cmd_fake_live(opts):
    """Calibrate + throw against a fake Motive (real UDP/NatNet) and a fake Spot."""
    from throw_trajectory import cubic_joint_trajectory
    from twin.live import LiveSource

    traj = trajectory_from(opts) or cubic_joint_trajectory(*FORWARD_THROW, opts.duration)
    with fake_lab(opts, traj[1][0]) as (robot, client, T, rb_id, _):
        live = LiveSource(robot, client, T, rb_id, ball=opts.ball, fly_time=1.5)

        def work(view):
            for i in range(opts.n):
                rec = live.throw(opts.duration, opts.release_fraction, target=opts.target, trajectory=traj)
                a = save_record(rec, "fakelive", opts, i)
                if view is not None:
                    view.post_result(rec, a, None, live.last_T_world_robot)

        with_optional_view(opts, robot, client, T, rb_id, opts.target, work)


def cmd_optimize(opts):
    from twin import optimizer as O
    from twin.live import LiveSource

    if opts.resume:
        opt = O.Optimizer.load(opts.resume)
        history_path = Path(opts.resume)
        print(f"resuming {history_path}: {len(opt.history)} throws so far")
    else:
        opt = O.Optimizer(target=tuple(opts.target), space=O.ThrowSpace(max_vel=opts.max_vel, max_acc=opts.max_acc))
        history_path = DATA_DIR / f"opt_{opts.source}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    DATA_DIR.mkdir(exist_ok=True)
    stop_at = opts.stop_at / 100.0

    def finish():
        best = opt.best()
        if best:
            print(f"\nBEST: miss {best['miss'] * 100:.1f} cm, "
                  + ", ".join(f"{k}={v:.3f}" for k, v in best["params"].items()))
        plot = O.save_plot(opt, history_path.with_suffix(".png"))
        print(f"history: {history_path}" + (f"\nplot:    {plot}" if plot else ""))

    if opts.source == "sim":
        src = SimSource(ball=opts.ball, optitrack=FakeOptiTrack())

        def thrower(duration, rf, traj, target):
            return src.throw(duration, rf, target=target, trajectory=traj)

        O.run(opt, thrower, opts.n, history_path, DATA_DIR, "opt_sim", stop_at=stop_at)
        return finish()

    lab = fake_lab(opts, opt.space.to_throw(np.mean(opt.space.bounds(), axis=0))[2][1][0]) \
        if opts.source == "fake-live" else real_lab(opts)
    with lab as (robot, client, T, rb_id, ctl):
        live = LiveSource(robot, client, T, rb_id, ball=opts.ball, fly_time=1.5)

        def work(view):
            def thrower(duration, rf, traj, target):
                if ctl is not None:
                    prepare_real_throw(ctl, traj, duration, f"[round {len(opt.history) + 1}]")
                return live.throw(duration, rf, target=target, trajectory=traj)

            def on_result(rec, a, entry):
                if view is not None:
                    view.post_result(rec, a, entry, live.last_T_world_robot)

            O.run(opt, thrower, opts.n, history_path, DATA_DIR, f"opt_{opts.source}", on_result=on_result,
                  stop_at=stop_at)

        with_optional_view(opts, robot, client, T, rb_id, opt.target, work)
    finish()


def cmd_plan(opts):
    from twin.click_planner import ClickPlanner, open_window
    from twin.optimizer import ThrowSpace
    from twin.planner import ThrowPlanner

    cid = open_window()  # must be PyBullet's first connection, see open_window()
    print("preparing the planner (standing pose, reachable area)...")
    # Search the whole joint range (shoulder yaw = aim all around, any swing
    # end, any timing); only the joint speed/acceleration limits and an
    # arm-body clearance check (ThrowPlanner.min_clearance) restrict it.
    from throw_trajectory import JOINT_LIMIT_MARGIN, JOINT_LIMITS
    jl = JOINT_LIMITS + np.array([JOINT_LIMIT_MARGIN, -JOINT_LIMIT_MARGIN])
    space = ThrowSpace(lower=(0.4, 0.05, jl[1, 0], jl[0, 0]), upper=(3.0, 0.95, jl[1, 1], jl[0, 1]),
                       max_vel=opts.max_vel, max_acc=opts.max_acc)
    planner = ThrowPlanner(space=space, gripper_delay=opts.gripper_delay, ball=opts.ball)
    (rmin, rmax), _ = planner.reach()
    print(f"reachable: {rmin:.2f}..{rmax:.2f} m from the shoulder axis, aim all around. "
          f"Left-click anywhere in the window.")
    try:
        ClickPlanner(planner, cid=cid, fast_fallback=opts.fast_fallback).run()
    finally:
        planner.close()


def cmd_fast_plan(opts):
    """Offline: plan a fast throw (windup / coast / brake) to --target and export it."""
    from twin.fast_throw import FastThrowPlanner, FastThrowSpace, save_fast_plan

    space = FastThrowSpace(max_vel=opts.max_vel, max_acc=opts.max_acc, gripper_delay=opts.gripper_delay)
    fp = FastThrowPlanner(space, ball=opts.ball)
    try:
        print(f"planning a throw to {opts.target} (joint limit {opts.max_vel} rad/s, "
              f"release at least {opts.min_speed} m/s)...")
        plan = fp.plan(opts.target, min_release_speed=opts.min_speed, verbose=True)
    finally:
        fp.close()
    if plan is None:
        sys.exit("No throw found within the limits (try a closer target or a higher --max-vel).")
    path = DATA_DIR / "plans" / time.strftime("fastplan_%Y%m%d_%H%M%S.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fast_plan(plan, space, path)
    print(f"  swing {plan['duration']:.2f} s, release {plan['release_speed']:.2f} m/s at "
          f"{plan['elevation_deg']:.0f} deg from {plan['release_height']:.2f} m, "
          f"peak joint speed {plan['peak_joint_speed']:.2f} rad/s, arm-body clearance "
          f"{plan['clearance'] * 100:.0f} cm")
    print(f"  physics lands {np.round(plan['sim_landing'], 3)} (target {np.round(plan['target'], 3)})")
    print(f"exported {path}\n  simulate:  ./spot sim\n  on Spot:   ./spot throw   (both use the newest plan)")


# Fixed cubic throw: overhand, straight ahead. Only the swing time and the
# gripper release time are free (./spot cubic --time T --release R).
CUBIC_START = [0.0, -2.2, 3.0, 0.0, 0.0, 1.57]
CUBIC_END = [0.0, -1.24, 1.8, 0.0, 0.0, 1.57]


def cmd_cubic(opts):
    """Offline: fixed start -> end cubic swing; set the time and the release, export a plan."""
    import json
    from throw_trajectory import MAX_JOINT_ACC, MAX_JOINT_VEL, check_trajectory, cubic_joint_trajectory
    T, R = opts.time, opts.release
    src = opts.trajectory_from or max(glob.glob(str(DATA_DIR / "plans" / "plan_*.json")),
                                      key=lambda f: Path(f).stat().st_mtime, default=None)
    start, end = CUBIC_START, CUBIC_END
    if src:  # the cubic trajectory used before: same start / end poses, new timing
        old = json.loads(Path(src).read_text())
        start, end = old["start_pose"], old["end_pose"]
        print(f"trajectory (start/end poses) from {src}")
    if T <= 0 or not (0.0 < R < T):
        sys.exit(f"--release must be between 0 and --time ({T} s).")
    traj = cubic_joint_trajectory(start, end, T)
    d = np.abs(np.subtract(end, start))
    print(f"cubic swing {T:.3f} s, gripper open sent at {R:.3f} s ({R / T:.0%} of the swing)")
    print(f"  peak joint speed {1.5 * d.max() / T:.2f} rad/s (limit {MAX_JOINT_VEL}), "
          f"peak acceleration {6 * d.max() / T**2:.1f} rad/s^2 (limit {MAX_JOINT_ACC})")
    problems = check_trajectory(*traj, MAX_JOINT_VEL, MAX_JOINT_ACC)
    if problems:
        sys.exit("Too fast: " + "; ".join(problems) + "  -> use a longer --time")
    path = DATA_DIR / "plans" / time.strftime("cubic_%Y%m%d_%H%M%S.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "target": None, "duration": T, "release_fraction": R / T,
        "start_pose": [float(v) for v in start], "end_pose": [float(v) for v in end], "sim_landing": None,
        "gripper_delay": opts.gripper_delay,
        "space": {"max_vel": MAX_JOINT_VEL, "max_acc": MAX_JOINT_ACC},
    }, indent=2))
    print(f"exported {path}\n  simulate:  ./spot sim\n  on Spot:   ./spot throw   (both use the newest plan)")


def latest_plan():
    files = glob.glob(str(DATA_DIR / "plans" / "*.json"))
    if not files:
        sys.exit("No plan yet. Make one first: python run_twin.py plan  (or fast-plan --target X Y)")
    return max(files, key=lambda f: Path(f).stat().st_mtime)


def plan_max_vel(path):
    """Joint speed limit the plan was made for (fast plans store it), else the lab default."""
    import json
    from throw_trajectory import MAX_JOINT_VEL
    return float(json.loads(Path(path).read_text()).get("space", {}).get("max_vel", MAX_JOINT_VEL))


def cmd_sim(opts):
    """Short: simulate a plan (default: the newest) and open the 3D replay."""
    opts.plan = opts.plan or latest_plan()
    opts.n, opts.target, opts.start_pose, opts.end_pose = 1, None, None, None
    opts.ball, opts.rate, opts.noise, opts.dropout = "tennis", 120.0, 0.001, 0.02
    from twin.planner import load_plan
    opts.gripper_delay = load_plan(opts.plan)[0]["gripper_delay"]  # simulate the delay the plan assumed
    cmd_simulate(opts)
    if not opts.no_view:
        Viewer([str(opts.last_record)]).run()


def motive_answers(opts, timeout=3.0):
    from twin.natnet import NatNetClient
    try:
        NatNetClient(opts.motive_ip, opts.local_ip, data_port=opts.data_port).start(timeout=timeout).stop()
        return True
    except (TimeoutError, OSError):
        return False


def cmd_throw(opts):
    """Short: throw a plan (default: the newest) on Spot with the lab settings.
    OptiTrack is not needed. With --record it records the throw with
    OptiTrack and shows the live 3D window (Motive must be streaming)."""
    opts.plan = opts.plan or latest_plan()
    for k, v in LAB.items():
        setattr(opts, k, v)
    opts.multicast, opts.external_estop, opts.calib = False, True, str(CALIB_FILE)
    opts.max_vel = plan_max_vel(opts.plan) if opts.max_vel is None else opts.max_vel
    from throw_trajectory import MAX_JOINT_ACC
    opts.max_acc, opts.n, opts.ball = MAX_JOINT_ACC, 1, "tennis"
    opts.target, opts.start_pose, opts.end_pose = None, None, None
    opts.duration, opts.release_fraction = 1.0, 0.5  # replaced by the plan
    print(f"Spot {opts.hostname}, joint limit {opts.max_vel} rad/s, e-stop: tablet / hardware")
    if opts.dry_run:
        from throw_trajectory import check_trajectory
        traj = trajectory_from(opts)
        problems = check_trajectory(*traj, opts.max_vel, opts.max_acc)
        print(f"start pose {np.round(traj[1][0], 3).tolist()}, peak joint speed "
              f"{np.abs(traj[2]).max():.2f} rad/s, limits: {'OK' if not problems else problems}")
        if opts.record:
            print(f"OptiTrack (--record): {'answers' if motive_answers(opts) else 'NOT answering -> the throw would stop'}")
        else:
            print("OptiTrack: not used (add --record to record the throw)")
        print("dry run: nothing sent to Spot.")
        return
    if opts.record:
        opts.live_view = not opts.no_view
        return cmd_live(opts)
    print("OptiTrack not used: the throw is not recorded (add --record to record it).")
    traj = trajectory_from(opts)
    with spot_session(opts) as ctl:
        ctl.power_on()
        ctl.stand()
        prepare_real_throw(ctl, traj, opts.duration, "[1/1]")
        ctl.throw(T=opts.duration, max_vel=opts.max_vel, max_acc=opts.max_acc, release=True,
                  release_fraction=opts.release_fraction, trajectory=traj)
        time.sleep(1.0)
        ctl.stow_arm()


def cmd_sit(opts):
    """LAB: leave Spot free for other programs. If it is still powered on
    (e.g. a throw was cut off), stow the arm, sit down and power off; then
    return the lease. Does nothing else; if another program (tablet) holds
    the lease it only says so."""
    from bosdyn.client.lease import ResourceAlreadyClaimedError
    opts.hostname, opts.external_estop = LAB["hostname"], True
    print(f"Spot {opts.hostname}: sit down, power off, release the lease...")
    try:
        with spot_session(opts) as ctl:
            if not ctl._robot.is_powered_on():
                print("Spot is already powered off (sitting).")
    except ResourceAlreadyClaimedError:
        print("Another program (e.g. the tablet) holds the lease: nothing done, Spot is theirs.")


def cmd_analyze(opts):
    for f in expand(opts.files):
        print(f)
        print_analysis(ThrowRecord.load(f))


def cmd_view(opts):
    files = expand(opts.files)
    if not files:
        print("No records found. Make one first: python run_twin.py simulate")
        return
    Viewer(files).run()


def expand(patterns):
    if not patterns:
        return sorted(glob.glob(str(DATA_DIR / "*.npz")))
    return sorted(f for pat in patterns for f in glob.glob(pat))


# ---- lab commands ----------------------------------------------------------

def cmd_setup_fixed(opts):
    """No robot commands: Spot stays where it is. Reads the body rigid body
    and a ball placed on Spot's forward centre line, saves the calibration."""
    from twin import calibration
    from twin.live import _rb_pose, _stable_points

    if not opts.local_ip:
        sys.exit("setup-fixed needs --local-ip (this PC's IP on the SpotRoom network).")
    client = mocap_client(opts, opts.up_axis)
    try:
        input("Place the ball about 1 m straight in front of Spot, on its centre line, "
              "high enough for the cameras (0.5-0.8 m, e.g. on a box). Press Enter...")
        t0 = time.time()
        time.sleep(1.0)
        frames = client.frames_between(t0, time.time())
        T_world_rb = _rb_pose(frames, opts.body_rb)
        if T_world_rb is None:
            ids = sorted({k for f in frames for k in f.rigid_bodies})
            sys.exit(f"Rigid body {opts.body_rb} not tracked (seen: {ids}). Check Motive.")
        origin = T_world_rb[:3, 3]
        pts = _stable_points([f.unlabeled() for f in frames])
        far = [p for p in pts if np.linalg.norm(p[:2] - origin[:2]) > 0.4]
        print(f"body rigid body at {np.round(origin, 3)}; unlabeled points: "
              f"{[np.round(p, 3).tolist() for p in pts]}")
        if len(far) != 1:
            sys.exit(f"Expected exactly one point more than 0.4 m from the body (the ball), found {len(far)}.")
        ball = far[0]
        T = calibration.fixed_body_from_heading_point(T_world_rb, ball)
        calibration.save(opts.out, T, opts.body_rb, opts.up_axis, 0.0)
        yaw = np.degrees(np.arctan2(*(ball[:2] - origin[:2])[::-1]))
        print(f"ball at {np.round(ball, 3)}, {np.linalg.norm(ball[:2] - origin[:2]):.2f} m away; "
              f"Spot heading {yaw:.1f} deg in Motive")
        print(f"saved {opts.out}. Don't move Spot; redo this if it moves.")
    finally:
        client.stop()


def cmd_calibrate(opts):
    from twin import calibration
    from twin.live import SpotRobotInterface, run_calibration

    client = mocap_client(opts, opts.up_axis)
    try:
        time.sleep(1.0)  # one frame right after connecting can be empty
        recent = client.frames_between(time.time() - 1.0, time.time())
        if not any(opts.body_rb in f.rigid_bodies for f in recent):
            ids = sorted({k for f in recent for k in f.rigid_bodies})
            sys.exit(f"Rigid body {opts.body_rb} not streamed (seen: {ids}). Check Motive.")
        with spot_session(opts) as ctl:
            ctl.power_on()
            ctl.stand()
            ctl.open_gripper()
            input("Put the ball in the gripper, then press Enter to close it...")
            ctl.close_gripper()
            time.sleep(1.0)
            input(f"The arm will move SLOWLY through {len(calibration.CALIBRATION_POSES)} poses. "
                  f"Clear the area, then press Enter (Ctrl+C aborts)...")
            debug = Path(opts.out).with_suffix(".debug.npz")
            print(f"raw calibration data is saved to {debug} after every pose")
            T, rms, data = run_calibration(SpotRobotInterface(ctl), client, opts.body_rb, save_path=debug)
            ctl.stow_arm()
        calibration.save(opts.out, T, opts.body_rb, opts.up_axis, rms)
        print(f"saved {opts.out}: residual {rms * 1000:.1f} mm "
              f"({'good' if rms < 0.005 else 'CHECK: should be a few mm'})")
    finally:
        client.stop()


def cmd_live(opts):
    from twin.live import LiveSource

    traj = trajectory_from(opts)
    if traj is None:
        import spot_throw_safe as real
        traj = real.sample_trajectory(opts.duration)
    with real_lab(opts) as (robot, client, T, rb_id, ctl):
        live = LiveSource(robot, client, T, rb_id, ball=opts.ball)
        live.raw_path = DATA_DIR / "last_live_raw.npz"

        def work(view):
            for i in range(opts.n):
                prepare_real_throw(ctl, traj, opts.duration, f"[{i + 1}/{opts.n}]")
                rec = live.throw(opts.duration, opts.release_fraction, target=opts.target, trajectory=traj)
                a = save_record(rec, "live", opts, i)
                if view is not None:
                    view.post_result(rec, a, None, live.last_T_world_robot)

        with_optional_view(opts, robot, client, T, rb_id, opts.target, work)


# ---- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def throw_args(p, default_duration=1.0):
        p.add_argument("--duration", type=float, default=default_duration, help="swing duration (s)")
        p.add_argument("--release-fraction", type=float, default=0.5)
        p.add_argument("--start-pose", type=float, nargs=6, help="custom trajectory: start joints sh0..wr1")
        p.add_argument("--end-pose", type=float, nargs=6, help="custom trajectory: end joints sh0..wr1")
        p.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"),
                       help="target on the floor, robot frame (m)")
        p.add_argument("--ball", default="tennis")
        p.add_argument("--n", type=int, default=1, help="number of throws")
        p.add_argument("--plan", help="use a plan exported by 'run_twin.py plan' (twin_data/plans/*.json)")

    def lab_args(p, hostname_positional=True):
        if hostname_positional:
            p.add_argument("hostname", help="Spot IP (SpotRoom: 10.0.0.30)")
        else:
            p.add_argument("--hostname", help="Spot IP (SpotRoom: 10.0.0.30), for --source live")
        p.add_argument("--motive-ip", default="10.0.0.112")
        p.add_argument("--local-ip", required=hostname_positional,
                       help="this PC's IP on the SpotRoom network (ipconfig)")
        p.add_argument("--multicast", action="store_true", help="Motive set to multicast (lab uses unicast)")
        p.add_argument("--data-port", type=int, default=1511, help="Motive streaming Data Port")
        p.add_argument("--external-estop", action="store_true", help="tablet / hardware e-stop is the authority")
        p.add_argument("--calib", default=str(CALIB_FILE))
        p.add_argument("--max-vel", type=float, default=2.5, help="joint velocity limit (rad/s), lab example: 2.5")
        p.add_argument("--max-acc", type=float, default=10.0, help="joint acceleration limit (rad/s^2)")

    view_help = "real-time 3D window while throwing"

    s = sub.add_parser("simulate", help="simulate throws and save records")
    throw_args(s)
    s.add_argument("--gripper-delay", type=float, default=0.1)
    s.add_argument("--rate", type=float, default=120.0, help="fake OptiTrack rate (Hz)")
    s.add_argument("--noise", type=float, default=0.001, help="fake OptiTrack noise std (m)")
    s.add_argument("--dropout", type=float, default=0.02, help="fraction of lost OptiTrack frames")
    s.set_defaults(func=cmd_simulate)

    f = sub.add_parser("fake-live", help="run the live pipeline against a fake Motive + fake Spot")
    throw_args(f)
    f.add_argument("--live-view", action="store_true", help=view_help)
    f.set_defaults(func=cmd_fake_live)

    o = sub.add_parser("optimize", help="Bayesian optimization of the throw to hit --target")
    o.add_argument("--source", choices=["sim", "fake-live", "live"], default="sim")
    o.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"), default=(0.95, 0.0),
                   help="target on the floor, robot frame (m)")
    o.add_argument("--n", type=int, default=20, help="max number of throws")
    o.add_argument("--stop-at", type=float, default=2.0, help="stop when the miss is below this (cm)")
    o.add_argument("--resume", help="continue a saved history (twin_data/opt_*.json)")
    o.add_argument("--ball", default="tennis")
    o.add_argument("--live-view", action="store_true", help=view_help)
    lab_args(o, hostname_positional=False)
    o.set_defaults(func=cmd_optimize)

    pl = sub.add_parser("plan", help="click a target on the floor -> throw trajectory and joint angles")
    pl.add_argument("--max-vel", type=float, default=2.5, help="joint velocity limit (rad/s)")
    pl.add_argument("--max-acc", type=float, default=10.0, help="joint acceleration limit (rad/s^2)")
    pl.add_argument("--gripper-delay", type=float, default=0.1, help="open command -> ball free (s)")
    pl.add_argument("--ball", default="tennis")
    pl.add_argument("--no-fast-fallback", dest="fast_fallback", action="store_false",
                    help="don't plan a fast throw when the gentle throw can't reach a click")
    pl.set_defaults(func=cmd_plan)

    fp = sub.add_parser("fast-plan", help="offline: plan a fast throw (windup/coast/brake) to a target")
    fp.add_argument("--target", type=float, nargs=2, required=True, metavar=("X", "Y"),
                    help="target on the floor, robot frame (m)")
    fp.add_argument("--max-vel", type=float, default=2.5, help="joint speed limit (rad/s)")
    fp.add_argument("--max-acc", type=float, default=10.0, help="joint acceleration limit (rad/s^2)")
    fp.add_argument("--min-speed", type=float, default=1.5, help="minimum ball release speed (m/s)")
    fp.add_argument("--gripper-delay", type=float, default=0.1)
    fp.add_argument("--ball", default="tennis")
    fp.set_defaults(func=cmd_fast_plan)

    a = sub.add_parser("analyze", help="print the analysis of saved records")
    a.add_argument("files", nargs="*")
    a.set_defaults(func=cmd_analyze)

    v = sub.add_parser("view", help="replay records in the PyBullet window")
    v.add_argument("files", nargs="*", help="default: all records in twin_data/")
    v.set_defaults(func=cmd_view)

    sf = sub.add_parser("setup-fixed", help="LAB, no robot commands: Spot fixed; heading from a ball in front")
    lab_args(sf, hostname_positional=False)
    sf.add_argument("--body-rb", type=int, required=True, help="Motive rigid body id of Spot's body")
    sf.add_argument("--up-axis", choices=["y", "z"], default="z")
    sf.add_argument("--out", default=str(CALIB_FILE))
    sf.set_defaults(func=cmd_setup_fixed)

    c = sub.add_parser("calibrate", help="LAB: calibrate the body markers (ball held in the gripper)")
    lab_args(c)
    c.add_argument("--body-rb", type=int, required=True, help="Motive rigid body id of Spot's body")
    c.add_argument("--up-axis", choices=["y", "z"], default="z", help="Motive streaming up axis")
    c.add_argument("--out", default=str(CALIB_FILE))
    c.set_defaults(func=cmd_calibrate)

    lv = sub.add_parser("live", help="LAB: throw with Spot and record with OptiTrack")
    lab_args(lv)
    throw_args(lv, default_duration=2.0)
    lv.add_argument("--live-view", action="store_true", help=view_help)
    lv.set_defaults(func=cmd_live)

    sm = sub.add_parser("sim", help="short: simulate the newest plan (or PLAN) and replay it in 3D")
    sm.add_argument("plan", nargs="?", help="plan file (default: newest in twin_data/plans)")
    sm.add_argument("--no-view", action="store_true", help="no 3D replay window")
    sm.set_defaults(func=cmd_sim)

    th = sub.add_parser("throw", help="LAB, short: throw the newest plan (or PLAN) on Spot, lab settings built in")
    th.add_argument("plan", nargs="?", help="plan file (default: newest in twin_data/plans)")
    th.add_argument("--max-vel", type=float, help="joint speed limit (default: the plan's, else 10)")
    th.add_argument("--record", action="store_true",
                    help="record with OptiTrack + live 3D window (Motive must be streaming)")
    th.add_argument("--no-view", action="store_true", help="with --record: no real-time 3D window")
    th.add_argument("--dry-run", action="store_true", help="show what would run; nothing is sent to Spot")
    th.set_defaults(func=cmd_throw)

    cu = sub.add_parser("cubic", help="offline: fixed cubic throw; set swing time and release time")
    cu.add_argument("--time", type=float, required=True, help="start -> end pose time (s)")
    cu.add_argument("--release", type=float, required=True, help="gripper open sent at this time after start (s)")
    cu.add_argument("--gripper-delay", type=float, default=0.1, help="simulation: open command -> ball free (s)")
    cu.add_argument("--trajectory-from", help="plan whose start/end poses to use (default: newest plan_*.json)")
    cu.set_defaults(func=cmd_cubic)

    st = sub.add_parser("sit", help="LAB: Spot sits down, powers off and releases the lease")
    st.set_defaults(func=cmd_sit)

    opts = parser.parse_args()
    opts.func(opts)


if __name__ == "__main__":
    main()
