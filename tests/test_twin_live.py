"""Tests for the live digital-twin pipeline, without robot or Motive.

  python tests/test_twin_live.py          (from D:\\spot_throw_code)

1. NatNet packets: encode -> parse round trip for NatNet 3.1 / 4.0 / 4.1, Y-up.
2. Calibration math recovers a known body-marker offset.
3. Calibration arm poses: within limits, gripper in front of the body, no
   arm/body collision.
4. End to end over real UDP sockets: fake Motive + fake Spot -> NatNet client
   -> calibration -> LiveSource throw -> ThrowRecord -> analysis, compared
   with the simulation's ground truth.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from throw_trajectory import JOINT_LIMITS, cubic_joint_trajectory  # noqa: E402
from twin import ballistics, calibration, natnet  # noqa: E402
from twin.geometry import flatten, inv_T, make_T, mat_to_quat, quat_to_mat, rot_z, transform_points  # noqa: E402

THROW = cubic_joint_trajectory([0, -2.2, 3.0, 0, 0, 1.57], [0, -1.2, 1.8, 0, 0, 1.57], 1.0)


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def test_natnet_roundtrip():
    rng = np.random.default_rng(1)
    rbs = [(3, rng.normal(size=3), mat_to_quat(rot_z(0.4))), (7, rng.normal(size=3), np.array([0, 0, 0, 1.0]))]
    markers = [(rng.normal(size=3), 0.014, 3, 1), (rng.normal(size=3), 0.067, 0, 0), (rng.normal(size=3), 0.01, 0, 9)]
    for version in [(3, 1), (4, 0), (4, 1), (4, 2)]:
        pkt = natnet.encode_frame(42, rbs, markers, *version, timestamp=12.5)
        f = natnet.parse_frame(pkt[4:], *version)
        assert f.frame_number == 42 and abs(f.timestamp - 12.5) < 1e-9, version
        assert sorted(f.rigid_bodies) == [3, 7], version
        assert np.allclose(f.rigid_bodies[3].pos, rbs[0][1], atol=1e-6), version
        assert np.allclose(f.rigid_bodies[3].quat, rbs[0][2], atol=1e-6), version
        assert len(f.markers) == 3 and [m.model_id for m in f.markers] == [3, 0, 0], version
        assert len(f.unlabeled()) == 2, version
        assert np.allclose(f.unlabeled()[0], markers[1][0], atol=1e-6), version
    # Y-up -> Z-up: a point 1 m above the floor in Y-up must end up at z = 1.
    pkt = natnet.encode_frame(0, [(1, np.array([0.2, 1.0, 0.3]), np.array([0, 0, 0, 1.0]))],
                              [(np.array([0.2, 1.0, 0.3]), 0.05, 0, 0)], 4, 1)
    f = natnet.to_z_up(natnet.parse_frame(pkt[4:], 4, 1))
    assert np.allclose(f.markers[0].pos, [0.2, -0.3, 1.0], atol=1e-6)
    # Identity in the Y-up world = axes of the Y-up world, seen from the Z-up world.
    assert np.allclose(quat_to_mat(f.rigid_bodies[1].quat), natnet.Y_UP_TO_Z_UP, atol=1e-6)
    assert np.allclose(f.rigid_bodies[1].pos, [0.2, -0.3, 1.0], atol=1e-6)
    print("1. NatNet round trip (3.1, 4.0, 4.1, 4.2, Y-up): OK")


def test_calibration_math():
    rng = np.random.default_rng(2)
    T_rb_body = make_T((-0.1, 0.05, 0.12), mat_to_quat(rot_z(0.2)))
    ball_body = rng.uniform([0.4, -0.4, 0.0], [1.0, 0.4, 0.6], (8, 3))
    T_world_rb = [make_T(rng.normal(size=3), mat_to_quat(rot_z(rng.uniform(-3, 3)))) for _ in range(8)]
    ball_world = np.array([transform_points(T @ T_rb_body, b[None])[0] for T, b in zip(T_world_rb, ball_body)])
    ball_world += rng.normal(0, 0.001, ball_world.shape)
    T, rms = calibration.solve_rb_to_body(ball_body, ball_world, T_world_rb)
    pos_err = np.linalg.norm(T[:3, 3] - T_rb_body[:3, 3])
    ang_err = rot_err_deg(T[:3, :3], T_rb_body[:3, :3])
    assert pos_err < 0.003 and ang_err < 0.5 and rms < 0.003, (pos_err, ang_err, rms)
    print(f"2. Calibration math: offset error {pos_err * 1000:.1f} mm, {ang_err:.2f} deg: OK")


def test_calibration_poses():
    import pybullet as p
    from twin.kinematics import ARM_JOINTS, ArmFK

    fk = ArmFK()
    try:
        pts = []
        for q in calibration.CALIBRATION_POSES:
            assert np.all(np.array(q) > JOINT_LIMITS[:, 0] + 0.05) and np.all(np.array(q) < JOINT_LIMITS[:, 1] - 0.05), q
            g = fk.grasp_point(q)
            pts.append(g)
            for j, qj in zip(ARM_JOINTS, q):
                p.resetJointState(fk.robot, j, qj, physicsClientId=fk.cid)
            for a in (21, 22, 23, 24, 25):
                for b in range(2, 18):
                    close = p.getClosestPoints(fk.robot, fk.robot, 0.02, linkIndexA=a, linkIndexB=b,
                                               physicsClientId=fk.cid)
                    assert not close, f"pose {q}: arm link {a} within 2 cm of body link {b}"
            assert g[0] > 0.3 and g[2] > 0.0, f"pose {q}: gripper at {g}, not in front/above the body"
            from twin.kinematics import WRIST_LINK
            w = np.array(p.getLinkState(fk.robot, WRIST_LINK, computeForwardKinematics=True,
                                        physicsClientId=fk.cid)[4])
            up = (g - w)[2] / np.linalg.norm(g - w)
            assert up > 0.95, f"pose {q}: gripper not pointing up (z component {up:.2f}), ball would be hidden"
        pts = np.array(pts)
        spread = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
        assert spread[2] > 0.05, f"calibration points nearly coplanar: {spread}"
    finally:
        fk.close()
    print(f"3. Calibration poses: {len(pts)} poses, gripper up, no collisions, spread {np.round(spread, 2)} m: OK")


def test_end_to_end():
    from twin.fake_world import FakeMotive, FakeRobot, FakeWorld
    from twin.live import LiveSource, run_calibration
    from twin.sources import FakeOptiTrack, SimSource

    sim = SimSource(optitrack=FakeOptiTrack(seed=0))
    world = FakeWorld(sim, start_pose=THROW[1][0], up_axis="y")
    motive = FakeMotive(world, ip="127.0.0.1", command_port=15100, natnet_version=(4, 1, 0, 0)).start()
    client = natnet.NatNetClient("127.0.0.1", "127.0.0.1", command_port=15100, data_port=15111, up_axis="y")
    try:
        client.start()
        time.sleep(0.5)
        n = len(client.frames)
        assert n > 30, f"only {n} frames in 0.5 s"
        print(f"4a. NatNet over UDP: connected to '{client.server_name}' NatNet {client.version}, "
              f"{n} frames in 0.5 s: OK")

        robot = FakeRobot(world)
        T_cal, rms, _ = run_calibration(robot, client, world.rb_id, settle_s=0.2, average_s=0.3, verbose=False)
        pos_err = np.linalg.norm(T_cal[:3, 3] - world.T_rb_body[:3, 3])
        ang_err = rot_err_deg(T_cal[:3, :3], world.T_rb_body[:3, :3])
        assert pos_err < 0.005 and ang_err < 1.0, (pos_err, ang_err)
        print(f"4b. Calibration through the fake lab (stray reflections present): rms {rms * 1000:.1f} mm, "
              f"offset error {pos_err * 1000:.1f} mm / {ang_err:.2f} deg: OK")

        live = LiveSource(robot, client, T_cal, world.rb_id, fly_time=1.5)
        rec = live.throw(1.0, 0.5, target=(1.0, 0.0), trajectory=THROW)
        path = rec.save(ROOT / "twin_data" / "fake_live_test.npz")
        P = rec.params
        a = ballistics.analyze(rec, P["ball_radius"], P["ball_mass"], P["drag_coeff"])

        # Ground truth from the simulation, in the same robot frame.
        log = world.last_result["log"]
        T_world_robot = flatten(make_T(log["body_pos"][0], log["body_orn"][0]))
        truth = transform_points(inv_T(T_world_robot), world.last_result["landing"][1][None])[0]
        land_err = np.linalg.norm(a["meas_landing"][:2] - truth[:2])
        floor_err = abs(rec.floor_z - (-T_world_robot[2, 3]))
        rel_err = abs(a["release_t"] - world.last_result["release_state"][0])
        found = np.mean(~np.isnan(rec.ball_pos[:, 0]))
        assert land_err < 0.03, land_err
        assert floor_err < 0.01, floor_err
        assert rel_err < 0.05, rel_err
        print(f"4c. Live throw -> record ({len(rec.ball_t)} mocap frames, ball found in {found:.0%}, "
              f"{len(rec.robot_t)} joint samples), saved {path.name}")
        print(f"    landing measured {np.round(a['meas_landing'][:2], 3)} vs truth {np.round(truth[:2], 3)}: "
              f"error {land_err * 100:.1f} cm; floor error {floor_err * 1000:.1f} mm; "
              f"release time error {rel_err * 1000:.0f} ms: OK")
    finally:
        client.stop()
        motive.stop()


def test_optimizer():
    from twin import optimizer as O
    from twin.sources import SimSource

    # GP sanity: recovers a smooth 2-D function from 30 samples.
    rng = np.random.default_rng(5)
    X = rng.random((30, 2))
    f = lambda X: np.sin(3 * X[:, 0]) + (X[:, 1] - 0.5) ** 2  # noqa: E731
    gp = O.GaussianProcess().fit(X, f(X))
    Xt = rng.random((200, 2))
    mu, sd = gp.predict(Xt)
    rmse = np.sqrt(np.mean((mu - f(Xt)) ** 2))
    assert rmse < 0.1, rmse  # function spread ~0.35, 30 random samples

    # Closed loop in simulation: a reachable target within 15 throws.
    from twin.sources import FakeOptiTrack
    src = SimSource(optitrack=FakeOptiTrack(seed=4))
    opt = O.Optimizer(target=(0.95, 0.15), seed=2)
    out = ROOT / "twin_data" / "test_opt.json"
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        best = O.run(opt, lambda d, rf, tr, tg: src.throw(d, rf, target=tg, trajectory=tr), 15, out,
                     ROOT / "twin_data", "testopt", stop_at=0.03)
    resumed = O.Optimizer.load(out)
    assert len(resumed.history) == len(opt.history) and resumed.target == (0.95, 0.15)
    assert best["miss"] < 0.05, best
    print(f"5. Optimizer: GP rmse {rmse:.3f}; sim closed loop hit {best['miss'] * 100:.1f} cm from the target "
          f"in {len(opt.history)} throws; history saved/resumed: OK")


def test_live_view():
    from twin.fake_world import FakeMotive, FakeRobot, FakeWorld
    from twin.live import LiveSource, run_calibration
    from twin.live_view import LiveView
    from twin.sources import SimSource

    world = FakeWorld(SimSource(), start_pose=THROW[1][0], up_axis="z")
    motive = FakeMotive(world, command_port=15200).start()
    client = natnet.NatNetClient("127.0.0.1", "127.0.0.1", command_port=15200, data_port=15211).start()
    try:
        robot = FakeRobot(world)
        T, _, _ = run_calibration(robot, client, world.rb_id, settle_s=0.2, average_s=0.3, verbose=False)
        live = LiveSource(robot, client, T, world.rb_id, fly_time=1.5)
        view = LiveView(robot, client, T, world.rb_id, target=(1.0, 0.0), gui=False)
        heights = []
        orig_update = view.update

        def update():
            orig_update()
            if view.trail:
                heights.append(view.trail[-1][2])
        view.update = update

        def work(v):
            rec = live.throw(1.0, 0.5, target=(1.0, 0.0), trajectory=THROW)
            P = rec.params
            v.post_result(rec, ballistics.analyze(rec, P["ball_radius"], P["ball_mass"], P["drag_coeff"],
                                                  fit_drag=False), None, live.last_T_world_robot)

        view.run(work, max_seconds=60)
        frac = view.frames_with_ball / max(view.frames_shown, 1)
        assert view.results_shown == 1, view.results_shown
        assert frac > 0.6, frac
        assert heights and max(heights) - min(heights) > 0.3, (min(heights), max(heights))
        print(f"6. Live view (headless): {view.frames_shown} frames drawn, ball tracked in {frac:.0%}, "
              f"tracked height range {min(heights):.2f}..{max(heights):.2f} m, result shown: OK")
    finally:
        client.stop()
        motive.stop()


def test_click_planner():
    import pybullet as p
    from twin.click_planner import ClickPlanner, floor_hit, mouse_ray
    from twin.planner import ThrowPlanner, load_plan

    # Mouse ray: project known floor points to pixels, click them, get them back.
    w, h = 1024, 768
    view = p.computeViewMatrixFromYawPitchRoll((0.75, 0, -0.5), 2.6, -90, -55, 0, 2)
    proj = list(p.computeProjectionMatrixFOV(60, w / h, 0.01, 100))
    # The PyBullet window's own matrix: infinite far plane (singular, cannot be inverted).
    proj_window = list(proj)
    proj_window[10], proj_window[11], proj_window[14] = -1.0, -1.0, 0.0
    worst = 0.0
    for pr in (proj, proj_window):
        M = np.array(pr).reshape(4, 4).T @ np.array(view).reshape(4, 4).T
        for pt in [(0.9, 0.0, -0.5), (1.2, 0.4, -0.5), (0.5, -0.6, -0.5)]:
            c = M @ np.array([*pt, 1.0])
            mx, my = (c[0] / c[3] + 1) / 2 * w, (1 - c[1] / c[3]) / 2 * h
            hit = floor_hit(*mouse_ray(w, h, view, pr, mx, my), -0.5)
            worst = max(worst, np.linalg.norm(hit - pt))
    assert worst < 1e-6, worst

    # Click -> plan (headless: call click() as the mouse handler would).
    planner = ThrowPlanner()
    ui = ClickPlanner(planner, gui=False, fast_fallback=True)
    try:
        ui.click((0.95, 0.2))
        t0 = time.time()
        seen = []
        while time.time() - t0 < 60 and not (ui.plan is not None and ui.plan.get("sim_landing") is not None):
            if ui.pending:
                seen.append(ui.pending[0])
            ui._consume_pending()
            ui._animate()
            time.sleep(0.02)
        assert ui.plan and ui.plan["ok"], ui.status
        err = np.linalg.norm(ui.plan["sim_landing"] - (0.95, 0.2))
        assert err < 0.02, err
        import tempfile
        path = ui.export(tempfile.mkdtemp())
        d, traj = load_plan(path)
        assert np.allclose(traj[1][0], d["start_pose"]) and np.allclose(traj[1][-1], d["end_pose"])
        # Beyond the gentle throw: falls back to a fast throw, which exports and reloads.
        ui.plan = None
        ui.click((1.5, 0.0))
        t0 = time.time()
        while time.time() - t0 < 300 and (ui.busy or ui.pending):
            ui._consume_pending()
            ui._animate()
            time.sleep(0.02)
        assert ui.plan and ui.plan.get("kind") == "fast", ui.status
        fast_err = np.linalg.norm(ui.plan["sim_landing"] - (1.5, 0.0))
        assert fast_err < 0.03, fast_err
        fpath = ui.export(tempfile.mkdtemp())
        assert fpath.name.startswith("fastplan_")
        d, traj = load_plan(fpath)
        assert np.allclose(traj[1][0], ui.plan["start_pose"])
        ui.click((3.0, 0.0))  # too far even for the fast throw at 2.5 rad/s
        while ui.busy or ui.pending:
            ui._consume_pending()
            time.sleep(0.02)
        assert "No throw reaches" in ui.status, ui.status
    finally:
        p.disconnect(ui.cid)
        planner.close()
    print(f"7. Click planner: mouse ray exact ({worst:.0e} m); click (0.95, 0.2) -> plan, physics lands "
          f"{err * 100:.1f} cm away; stages {sorted(set(seen))}; export/load OK; 1.5 m -> fast throw "
          f"{fast_err * 100:.1f} cm away; 3 m click rejected: OK")
    return path


def test_fast_throw():
    """Windup / coast / brake throw: smooth profile, top speed while the
    gripper opens, plan file round trip, and a physics check of one plan."""
    import json
    import tempfile
    from twin.fast_throw import FastThrowPlanner, FastThrowSpace, save_fast_plan
    from twin.planner import load_plan
    from throw_trajectory import check_trajectory

    space = FastThrowSpace(max_vel=2.5)
    x = [-1.6, 1.4, -0.6, 2.0, -2.5, 1.0, 0.0]
    T, rf, (t, pos, vel) = space.build(x)
    assert t[0] == 0 and abs(t[-1] - T) < 1e-9
    assert np.allclose(vel[0], 0) and np.allclose(vel[-1], 0), "must start and end at rest"
    dv = np.abs(np.diff(vel, axis=0)).max()
    assert dv < 0.2, f"velocity jumps by {dv:.2f} rad/s between samples"
    # position = integral of velocity
    assert np.allclose(pos[-1] - pos[0], np.sum((vel[1:] + vel[:-1]) / 2 * np.diff(t)[:, None], axis=0), atol=2e-3)
    t_free = rf * T + space.gripper_delay
    ta = space.timing(np.array(x[3:6]))
    tc = space.coast_time()
    assert abs(t_free - (ta + tc)) < 1e-9, "ball free when braking starts"
    assert rf * T >= ta + space.open_margin - 1e-9, "open command only after the arm stops accelerating"
    assert not check_trajectory(t, pos, vel, 2.5, 10.0)

    fp = FastThrowPlanner(space)
    try:
        plan = fp.plan([1.5, 0.0], sim_iters=3, n_starts=24)
        assert plan is not None, "no fast throw to 1.5 m at 2.5 rad/s"
        assert np.linalg.norm(np.subtract(plan["sim_landing"], [1.5, 0.0])) < 0.03, plan["sim_landing"]
        assert plan["release_speed"] >= 1.45 and plan["release_height"] >= 1.0, plan
        assert plan["clearance"] >= space.clearance
        assert plan["rehit"] <= fp.max_rehit, "the gripper must not come back at the ball"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fast.json"
            save_fast_plan(plan, space, path)
            dd, traj = load_plan(path)
            assert json.loads(path.read_text())["kind"] == "fast"
            assert np.allclose(traj[1], fp.space.build(plan["x"])[2][1])
    finally:
        fp.close()
    print(f"8. Fast throw: smooth windup/coast/brake; 1.5 m plan: release {plan['release_speed']:.2f} m/s "
          f"at {plan['elevation_deg']:.0f} deg from {plan['release_height']:.2f} m, physics lands "
          f"{np.round(plan['sim_landing'], 3)}, clearance {plan['clearance'] * 100:.0f} cm: OK")


if __name__ == "__main__":
    test_natnet_roundtrip()
    test_calibration_math()
    test_calibration_poses()
    test_end_to_end()
    test_optimizer()
    test_live_view()
    test_click_planner()
    test_fast_throw()
    print("All tests passed.")
