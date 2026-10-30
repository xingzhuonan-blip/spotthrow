import numpy as np 
import pybullet as p 
from collections import namedtuple
np.set_printoptions(precision=3)


ball_param = namedtuple("ball_param", ('radius', 'mass', 'drag_coeff'))

def get_max_force_vel(robot_id, joint_id):
    id, name, type, q_ind, u_ind, flag, damp, friction, lower, upper, max_force, max_vel, link_name, *_ = p.getJointInfo(robot_id, joint_id)
    return max_force, max_vel


class Spot:
    initial_legs = (
        0.1, 0.9, -1.5,  # front left
        -0.1, 0.9, -1.5, # front right
        0.1, 1.1, -1.5,  # back left
        -0.1, 1.1, -1.5, # back right
    )
    initial_arm = (
        0.0, -3.14, 0, 3.14,      # arm0_sh0, arm0_sh1, arm0_el0
        0.0, 0.0, 0.0, 0 ,      # arm0_el1, arm0_wr0, arm0_wr1
    )
    # ping parameters: radius, mass, drag coeff
    ball_parameters = {
        "ping_pong": ball_param(radius=0.0200, mass=0.0027, drag_coeff=0.47),
        "tennis":    ball_param(radius=0.0335, mass=0.0570, drag_coeff=0.55),
        "golf":      ball_param(radius=0.02135, mass=0.0459, drag_coeff=0.25),
        "baseball":  ball_param(radius=0.0368, mass=0.1450, drag_coeff=0.35),
    }
    
    def __init__(self, ball_type, visualize=False, initial_sim_steps=25):
        self.visualize = visualize
        p.connect(p.GUI if visualize else p.DIRECT)
        plane = p.loadURDF('plane.urdf')
        self.spot_arm = p.loadURDF('fixed_spot/spot_fixed_base.urdf')
        # p.resetBasePositionAndOrientation(spot_arm, (0, 0, 0.0), (0,0,0,1))
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        for i in range(6, 18):
            p.resetJointState(self.spot_arm, i, self.initial_legs[i-6])
            max_force, max_vel = get_max_force_vel(self.spot_arm, i)
            p.setJointMotorControl2(self.spot_arm, i, controlMode=p.POSITION_CONTROL, targetPosition=self.initial_legs[i-6], maxVelocity=max_vel, force=max_force) # control them to be fixed
        for i in range(18, 26):
            p.resetJointState(self.spot_arm, i, self.initial_arm[i-18])
            max_force, max_vel = get_max_force_vel(self.spot_arm, i)
            p.setJointMotorControl2(self.spot_arm, i, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=max_force, maxVelocity=max_vel) # control them to be fixed
        for _ in range(initial_sim_steps):
            p.stepSimulation()
            
        self.ball_type = ball_type
        self._add_ball(ball_type)
            
    def finish(self):
        p.disconnect()
        
    def get_joint_state(self,):
        """Joint values and angular velocities (first 7 are values and last 7 are angular velocities)
        Returns joint_state: 
            _type_: np,Array
        """
        inds = [18, 19, 21, 22, 23, 24, 25]
        infos = p.getJointStates(self.spot_arm, inds)
        q = np.array([info[0] for info in infos])
        q_dot = np.array([info[1] for info in infos])
        return np.concatenate([q, q_dot])
        
    def move_shoulder_pos(self, pos, max_vel=None):
        ind = 19
        max_force, max_vel_nom = get_max_force_vel(self.spot_arm, ind)
        max_vel = max_vel_nom if max_vel is None else max_vel
        p.setJointMotorControl2(self.spot_arm, ind, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=max_force, maxVelocity=max_vel) 
    
    def move_elbow_pos(self, pos, max_vel=None):
        ind = 21
        max_force, max_vel_nom = get_max_force_vel(self.spot_arm, ind)
        max_vel = max_vel_nom if max_vel is None else max_vel
        p.setJointMotorControl2(self.spot_arm, ind, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=max_force, maxVelocity=max_vel) 

    def rotate_elbow(self, pos, max_vel=None):
        ind = 22
        max_force, max_vel_nom = get_max_force_vel(self.spot_arm, ind)
        max_vel = max_vel_nom if max_vel is None else max_vel
        p.setJointMotorControl2(self.spot_arm, ind, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=max_force, maxVelocity=max_vel) 

    def rotate_wrist(self, pos, max_vel=None):
        ind = 24
        max_force, max_vel_nom = get_max_force_vel(self.spot_arm, ind)
        max_vel = max_vel_nom if max_vel is None else max_vel
        p.setJointMotorControl2(self.spot_arm, ind, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=max_force, maxVelocity=max_vel) 

    def change_gripper(self, open_gripper=False):
        target = -1 if open_gripper else 0
        ind = 25
        max_force, max_vel = get_max_force_vel(self.spot_arm, ind)
        p.setJointMotorControl2(self.spot_arm, ind, controlMode=p.POSITION_CONTROL, targetPosition=target, force=max_force, maxVelocity=max_vel) 

    def get_ball_state(self, ):
        pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        vel, _ = p.getBaseVelocity(self.ball_id)
        return np.concatenate([pos, vel])

    def _add_ball(self, type='ping_pong'):
        radius, mass, _ = self.ball_parameters[type]
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius
        )
        colors = {
            'ping_pong': (1.0, 0.6, 0.0, 1.0), 
            'tennis': (0.75, 0.95, 0.10, 1.0),
            'golf': (0.98, 0.98, 0.98, 1.0), 
            'baseball': (0.0, 1.0, 0.0, 1.0)
            }
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=colors[type] 
        )
        self.ball_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=(0.54, -0.0, 0.75), # end effector position
        )
       
    
    # default values for ping pong ball
    def _add_air_force(self, ):
        obj_id = self.ball_id
        radius, mass, drag_coeff = self.ball_parameters[self.ball_type]
        linear_velocity, angular_velocity = p.getBaseVelocity(obj_id)
        v_squared = np.sum(np.square(linear_velocity))
        speed = np.sqrt(v_squared)
        rho = 1.225         # Air density (kg/m^3)
        area = np.pi*radius**2
        # Avoid division by zero when the object is stationary
        if speed > 0.001:
            v_hat = np.array(linear_velocity) / speed
            # Compute drag force magnitude: F = 0.5 * rho * Cd * A * v^2
            drag_force_mag = 0.5 * rho * drag_coeff * area * v_squared
            drag_force = -drag_force_mag * v_hat
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            p.applyExternalForce(
                objectUniqueId=obj_id,
                linkIndex=-1,
                forceObj=drag_force.tolist(),
                posObj=pos,
                flags=p.WORLD_FRAME
            )
    
    def simulate(self):
        self._add_air_force()
        p.stepSimulation()
        

def throw_ball(ball_type='ping_pong'):
    """Simulate spot throwing different balls

    Args:
        ball_type (str, optional): Defaults to 'ping_pong'.

    Returns:
        np.array: ball states
    """
    spot = Spot(ball_type=ball_type, visualize=False)
    spot.move_shoulder_pos(-2.5, 1)
    ball_states = []
    time_step = 0
    while time_step<400:
        time_step+=1
        spot.simulate()
        joint_state = spot.get_joint_state()
        ball_state = spot.get_ball_state()
        ball_states.append(ball_state)
        if np.abs((joint_state[1]+2.5))<0.05:
            spot.rotate_wrist(np.pi/2, max_vel=2)
        if np.abs(joint_state[5]-np.pi/2)<0.02:
            spot.move_elbow_pos(np.pi/2)
            spot.change_gripper(open_gripper=True)
    spot.finish()
    return np.array(ball_states)



def shoulder_traj(t, T=1, ):
    a0 = np.pi 
    a1 = 0 
    a2 = -1.5*np.pi/(T**2)
    a3 = np.pi/(T**3)
    return a0+a2*t**2+a3*t**3

def throw_ball_mp(T, ball_type='ping_pong'):
    spot = Spot(ball_type=ball_type, visualize=False)
    spot.move_shoulder_pos(-2.5, 1)
    ball_states = []
    time_step = 0
    rel_t = 0
    while rel_t<T+1:
        time_step+=1
        spot.simulate()
        joint_state = spot.get_joint_state()
        ball_state = spot.get_ball_state()
        ball_states.append(ball_state)
        if np.abs((joint_state[1]+2.5))<0.05:
            spot.rotate_wrist(np.pi/2, max_vel=2)
        if np.abs(joint_state[5]-np.pi/2)<0.02:
            
            spot.move_elbow_pos(shoulder_traj(rel_t))
            rel_t += 0.01
            spot.change_gripper(open_gripper=True)
    spot.finish()
    return np.array(ball_states)
 
if __name__ == '__main__':
    import time
    spot = Spot(ball_type='ping_pong', visualize=True)
    video_log_id = p.startStateLogging(
        p.STATE_LOGGING_VIDEO_MP4,
        "simulation_pingpong_t0.5.mp4"
    )
    spot.move_shoulder_pos(-2.5, 1)
    time_step = 0
    t_ref = 0
    while t_ref<5.5:
        time_step+=1
        spot.simulate()
        joint_state = spot.get_joint_state()
        ball_state = spot.get_ball_state()
        print(t_ref, ball_state)
        if np.abs((joint_state[1]+2.5))<0.05:
            spot.rotate_wrist(np.pi/2, max_vel=2)
        if np.abs(joint_state[5]-np.pi/2)<0.02:
            if t_ref<0.5:
                spot.move_elbow_pos(shoulder_traj(t_ref, 0.5)) 
            t_ref += 0.01
            if t_ref>0.20:
                spot.change_gripper(open_gripper=True)
        time.sleep(0.01)
    p.stopStateLogging(video_log_id)
    spot.finish()