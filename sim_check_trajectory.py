"""Replay the real-robot throw trajectory (spot_throw_safe.py) in PyBullet.

Use this BEFORE running on Spot to check what the robot would do:
  * joint limits and tracking of the commanded trajectory
  * arm collisions with the body / legs / ground
  * ball release speed and direction, and where it lands

The trajectory comes straight from spot_throw_safe.sample_trajectory, so this
checks exactly what would be sent to the robot.

Examples:
  python sim_check_trajectory.py --duration 1.0 --release-fraction 0.5
  python sim_check_trajectory.py --duration 1.0 --gui
"""

import argparse
import sys
import time
from unittest import mock

import numpy as np
import pybullet
from pybullet_utils import bullet_client

# spot_throw_safe imports the Spot SDK at module level. Only its pure
# trajectory functions are used here, so stub the SDK if it isn't installed.
try:
    import bosdyn.client  # noqa: F401
except ImportError:
    for _name in ["bosdyn", "bosdyn.client", "bosdyn.client.util", "bosdyn.client.estop",
                  "bosdyn.client.lease", "bosdyn.client.robot_command", "bosdyn.client.robot_state"]:
        sys.modules[_name] = mock.MagicMock()
    sys.modules["bosdyn.client"].ResponseError = type("ResponseError", (Exception,), {})
    sys.modules["bosdyn.client"].RpcError = type("RpcError", (Exception,), {})

import spot_throw_safe as real
from sim_spot_throwing import Spot
from throw_trajectory import check_trajectory

# SDK joint order (arm0.sh0 ... arm0.wr1) -> PyBullet joint index in the URDF.
ARM_JOINTS = [18, 19, 21, 22, 23, 24]
GRIPPER_JOINT = 25
WRIST_LINK = 24                          # arm_link_wr1 (carries the jaws)
BALL_OFFSET_IN_WRIST = (0.19, 0.0, -0.01)  # between the jaws, measured from the URDF
ARM_LINKS = [21, 22, 23, 24, 25]         # el0 onwards: links that could hit the body
BODY_LINKS = list(range(2, 18))          # body, rails, hips and legs
BODY_LINK = 2                            # URDF 'body' link = Spot's body frame

SIM_DT = 1.0 / 240.0
RHO_AIR = 1.225


def interp_traj(times, positions, velocities, t):
    """Linear interpolation of the sampled trajectory (holds the end pose)."""
    times = np.asarray(times)
    if t >= times[-1]:
        return np.asarray(positions[-1]), np.zeros(6)
    k = max(int(np.searchsorted(times, t, side="right")) - 1, 0)
    a = (t - times[k]) / (times[k + 1] - times[k])
    q = (1 - a) * np.asarray(positions[k]) + a * np.asarray(positions[k + 1])
    qd = (1 - a) * np.asarray(velocities[k]) + a * np.asarray(velocities[k + 1])
    return q, qd


def run(T, release_fraction, gripper_delay, ball_type, max_vel, max_acc, gui, fly_time,
        check_collisions=True, verbose=True, trajectory=None):
    """Simulate one throw. Returns a dict with the per-step log and key events.

    trajectory: optional (times, positions, velocities) to test a new design
    (e.g. throw_trajectory.cubic_joint_trajectory). Default: the trajectory
    spot_throw_safe.py would send to the robot."""
    if trajectory is None:
        times, positions, velocities = real.sample_trajectory(T)
        try:
            real.validate_trajectory(times, positions, velocities, max_vel, max_acc)
        except ValueError as err:
            if verbose:
                print(f"[REJECTED by spot_throw_safe validation] {err}")
                print("The real script would refuse this trajectory; simulating anyway for inspection.")
    else:
        times, positions, velocities = (np.asarray(x, float) for x in trajectory)
        problems = check_trajectory(times, positions, velocities, max_vel, max_acc)
        if problems and verbose:
            print("[TRAJECTORY CHECK FAILED] " + "; ".join(problems))
            print("Simulating anyway for inspection.")
    start_pose = np.asarray(positions[0], float)

    # Own client: plain pybullet calls go to client 0, which may belong to
    # someone else (e.g. the digital twin's kinematics) when run in-process.
    p = bullet_client.BulletClient(pybullet.GUI if gui else pybullet.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(SIM_DT)
    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF("fixed_spot/spot_fixed_base.urdf")

    # Legs fixed in a standing pose (same as sim_spot_throwing.Spot).
    for i in range(6, 18):
        p.resetJointState(robot, i, Spot.initial_legs[i - 6])
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, targetPosition=Spot.initial_legs[i - 6],
                                force=p.getJointInfo(robot, i)[10])
    # Arm starts where initial_move() would leave it.
    for j, q in zip(ARM_JOINTS, start_pose):
        p.resetJointState(robot, j, q)
    p.resetJointState(robot, GRIPPER_JOINT, 0.0)

    # Ball held rigidly between the jaws until release.
    radius, mass, drag_coeff = Spot.ball_parameters[ball_type]
    wrist_pos, wrist_orn = p.getLinkState(robot, WRIST_LINK)[4:6]
    ball_pos, _ = p.multiplyTransforms(wrist_pos, wrist_orn, BALL_OFFSET_IN_WRIST, (0, 0, 0, 1))
    ball = p.createMultiBody(mass, p.createCollisionShape(p.GEOM_SPHERE, radius=radius),
                             p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=(0.75, 0.95, 0.1, 1)),
                             basePosition=ball_pos)
    # PyBullet adds velocity damping to every body by default; the only air
    # resistance should be our explicit drag force.
    p.changeDynamics(ball, -1, linearDamping=0.0, angularDamping=0.0)
    for link in (WRIST_LINK, GRIPPER_JOINT):
        p.setCollisionFilterPair(robot, ball, link, -1, 0)
    # Constraint frames are relative to the link's centre of mass, not the URDF
    # link frame, so express the ball position in the COM frame.
    com_pos, com_orn = p.getLinkState(robot, WRIST_LINK)[0:2]
    inv_pos, inv_orn = p.invertTransform(com_pos, com_orn)
    ball_in_com, _ = p.multiplyTransforms(inv_pos, inv_orn, ball_pos, (0, 0, 0, 1))
    grasp = p.createConstraint(robot, WRIST_LINK, ball, -1, p.JOINT_FIXED, (0, 0, 0),
                               ball_in_com, (0, 0, 0))

    def command_arm(q, qd):
        for j, qj, qdj in zip(ARM_JOINTS, q, qd):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=qj, targetVelocity=qdj,
                                    maxVelocity=max_vel, force=p.getJointInfo(robot, j)[10])

    # Settle at the start pose, like the pause before the swing on the robot.
    command_arm(start_pose, np.zeros(6))
    for _ in range(int(0.5 / SIM_DT)):
        p.stepSimulation()
    # The robot is not fixed to the world; it settles on its legs. Record where.
    base_pose = p.getBasePositionAndOrientation(robot)

    release_cmd_t = release_fraction * T
    release_t = release_cmd_t + gripper_delay  # when the ball actually leaves
    released = False
    log = {"t": [], "q_cmd": [], "q": [], "qd": [], "ball": [], "body_pos": [], "body_orn": []}
    release_state = None
    min_clearance = np.inf
    clearance_pair = None
    arm_ground_contact = False
    ball_hit_robot = False
    landing = None

    n_steps = int((T + fly_time) / SIM_DT)
    for k in range(n_steps):
        t = k * SIM_DT
        q_cmd, qd_cmd = interp_traj(times, positions, velocities, t)
        command_arm(q_cmd, qd_cmd)
        if t >= release_cmd_t:
            p.setJointMotorControl2(robot, GRIPPER_JOINT, p.POSITION_CONTROL, targetPosition=-1.0,
                                    force=p.getJointInfo(robot, GRIPPER_JOINT)[10])
        if not released and t >= release_t:
            pos, _ = p.getBasePositionAndOrientation(ball)
            vel, _ = p.getBaseVelocity(ball)
            release_state = (t, np.array(pos), np.array(vel))
            p.removeConstraint(grasp)
            released = True

        # Aerodynamic drag on the ball once it is free.
        if released:
            vel = np.array(p.getBaseVelocity(ball)[0])
            speed = np.linalg.norm(vel)
            if speed > 1e-3:
                f = -0.5 * RHO_AIR * drag_coeff * np.pi * radius**2 * speed * vel
                p.applyExternalForce(ball, -1, f.tolist(), p.getBasePositionAndOrientation(ball)[0],
                                     p.WORLD_FRAME)
        p.stepSimulation()

        states = p.getJointStates(robot, ARM_JOINTS)
        bpos, _ = p.getBasePositionAndOrientation(ball)
        log["t"].append(t)
        log["q_cmd"].append(q_cmd)
        log["q"].append([s[0] for s in states])
        log["qd"].append([s[1] for s in states])
        log["ball"].append(bpos)
        log.setdefault("hand", []).append(p.getLinkState(robot, WRIST_LINK)[4])
        body_pos, body_orn = p.getLinkState(robot, BODY_LINK)[4:6]
        log["body_pos"].append(body_pos)
        log["body_orn"].append(body_orn)

        for a in ARM_LINKS:
            for b in (BODY_LINKS if check_collisions else []):
                pts = p.getClosestPoints(robot, robot, 0.05, linkIndexA=a, linkIndexB=b)
                for pt in pts:
                    if pt[8] < min_clearance:
                        min_clearance = pt[8]
                        clearance_pair = (a, b)
            # Only arm links count: the feet always touch the ground.
            if p.getContactPoints(robot, plane, linkIndexA=a):
                arm_ground_contact = True
        if released and t > release_t + 0.05 and p.getContactPoints(ball, robot):
            ball_hit_robot = True
        if released and landing is None and bpos[2] <= radius + 1e-3:
            landing = (t, np.array(bpos))

        if gui:
            time.sleep(SIM_DT)

    link_names = {i: p.getJointInfo(robot, i)[12].decode() for i in ARM_LINKS + BODY_LINKS}
    p.disconnect()
    pair_names = None if clearance_pair is None else tuple(link_names[i] for i in clearance_pair)
    if verbose:
        report(T, log, release_state, landing, min_clearance, pair_names, arm_ground_contact,
               ball_hit_robot, release_cmd_t, gripper_delay)
    return {"log": log, "release_state": release_state, "landing": landing,
            "release_cmd_t": release_cmd_t, "ball_radius": radius, "drag_coeff": drag_coeff,
            "ball_mass": mass, "base_pose": base_pose}


def report(T, log, release_state, landing, min_clearance, pair_names, arm_ground_contact, ball_hit_robot,
           release_cmd_t, gripper_delay):
    q = np.array(log["q"])
    q_cmd = np.array(log["q_cmd"])
    qd = np.array(log["qd"])
    print("\n=== Arm ===")
    ok = True
    for i, name in enumerate(real.ARM_JOINT_NAMES):
        lo, hi = real.JOINT_LIMITS[i]
        if q[:, i].min() < lo or q[:, i].max() > hi:
            print(f"  [FAIL] {name} left its limits: [{q[:, i].min():.3f}, {q[:, i].max():.3f}]")
            ok = False
    if ok:
        print("  Joint limits: OK")
    track_err = np.abs(q - q_cmd).max(axis=0)
    print(f"  Max tracking error per joint (rad): {np.round(track_err, 3)}")
    print(f"  Peak joint speed per joint (rad/s): {np.round(np.abs(qd).max(axis=0), 2)}")
    if min_clearance < 0.005:
        print(f"  [FAIL] Arm collides with body/legs: {pair_names} ({min_clearance*100:.1f} cm)")
    elif min_clearance < 0.05:
        print(f"  [WARN] Arm comes within {min_clearance*100:.1f} cm of body/legs: {pair_names}")
    else:
        print("  Arm-body clearance: > 5 cm, OK")
    print("  Arm touches ground: " + ("[FAIL] yes" if arm_ground_contact else "no, OK"))
    hand = np.array(log["hand"])
    idx = np.linspace(0, len(hand) - 1, 5).astype(int)
    print("  Wrist position over time (x forward, z up):")
    for i in idx:
        print(f"    t={log['t'][i]:.2f}s  {np.round(hand[i], 3)}")

    print("\n=== Ball ===")
    if release_state is None:
        print("  Never released.")
        return
    t_r, pos_r, vel_r = release_state
    speed = np.linalg.norm(vel_r)
    horiz = np.linalg.norm(vel_r[:2])
    elev = np.degrees(np.arctan2(vel_r[2], horiz))
    heading = np.degrees(np.arctan2(vel_r[1], vel_r[0]))
    print(f"  Gripper open command at t = {release_cmd_t:.2f} s, ball free at t = {t_r:.2f} s "
          f"(delay {gripper_delay*1000:.0f} ms)")
    i_r = int(np.searchsorted(log["t"], t_r))
    grip_gap = np.linalg.norm(np.array(log["ball"][i_r - 1]) - np.array(log["hand"][i_r - 1]))
    print(f"  Release position (m): {np.round(pos_r, 3)}  (ball-wrist distance {grip_gap:.3f} m, "
          f"expected {np.linalg.norm(BALL_OFFSET_IN_WRIST):.3f})")
    print(f"  Release speed: {speed:.2f} m/s, elevation {elev:.1f} deg, "
          f"heading {heading:.1f} deg (0 = straight ahead, 180 = backwards)")
    if ball_hit_robot:
        print("  [WARN] Ball hit the robot after release.")
    if landing is None:
        print("  Ball did not land within the simulated time.")
    else:
        t_l, pos_l = landing
        dist = np.linalg.norm(pos_l[:2] - pos_r[:2])
        print(f"  First ground contact at t = {t_l:.2f} s, position (m): {np.round(pos_l[:2], 3)}, "
              f"{dist:.2f} m from release point")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=3.0, help="Swing duration T (s), as in spot_throw_safe.")
    parser.add_argument("--release-fraction", type=float, default=0.5)
    parser.add_argument("--gripper-delay", type=float, default=0.1,
                        help="Time from open command to the ball actually leaving (s). Measure on the robot!")
    parser.add_argument("--ball", default="tennis", choices=list(Spot.ball_parameters))
    parser.add_argument("--max-vel", type=float, default=8.0)
    parser.add_argument("--max-acc", type=float, default=10.0)
    parser.add_argument("--fly-time", type=float, default=2.0, help="Extra simulated time after the swing (s).")
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet window (real-time).")
    opts = parser.parse_args()
    run(opts.duration, opts.release_fraction, opts.gripper_delay, opts.ball,
        min(opts.max_vel, real.HARD_MAX_JOINT_VEL), min(opts.max_acc, real.HARD_MAX_JOINT_ACC),
        opts.gui, opts.fly_time)


if __name__ == "__main__":
    main()
