
from f16_utils import GcasAutopilot, LowLevelController, get_state_names
from f16_sub_utils import subf16_model

import math

#TODO highlevel/controlled_f16.py
from math import sin, cos
import numpy as np
from numpy import deg2rad


def controlled_f16(t, x_f16, u_ref, llc, f16_model='morelli', v2_integrators=False):
    'returns the LQR-controlled F-16 state derivatives and more'

    assert isinstance(x_f16, np.ndarray)
    assert isinstance(llc, LowLevelController)
    assert u_ref.size == 4

    assert f16_model in ['stevens', 'morelli'], 'Unknown F16_model: {}'.format(f16_model)

    x_ctrl, u_deg = llc.get_u_deg(u_ref, x_f16)

    # Note: Control vector (u) for subF16 is in units of degrees
    xd_model, Nz, Ny, _, _ = subf16_model(x_f16[0:13], u_deg, f16_model)

    if v2_integrators:
        # integrators from matlab v2 model
        ps = xd_model[6] * cos(xd_model[1]) + xd_model[8] * sin(xd_model[1])

        Ny_r = Ny + xd_model[8]
    else:
        # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
        ps = x_ctrl[4] * cos(x_ctrl[0]) + x_ctrl[5] * sin(x_ctrl[0])

        # Calculate (side force + yaw rate) term
        Ny_r = Ny + x_ctrl[5]

    xd = np.zeros((x_f16.shape[0],))
    xd[:len(xd_model)] = xd_model

    # integrators from low-level controller
    start = len(xd_model)
    end = start + llc.get_num_integrators()
    int_der = llc.get_integrator_derivatives(t, x_f16, u_ref, Nz, ps, Ny_r)
    xd[start:end] = int_der

    # Convert all degree values to radians for output
    u_rad = np.zeros((7,)) # throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref

    u_rad[0] = u_deg[0] # throttle

    for i in range(1, 4):
        u_rad[i] = deg2rad(u_deg[i])

    u_rad[4:7] = u_ref[0:3] # inner-loop commands are 4-7

    return xd, u_rad, Nz, ps, Ny_r



#TODO run_f16_sim.py
import time
import numpy as np
from scipy.integrate import RK45


def get_extended_states(ap, t, full_state, model_str, v2_integrators):
    '''get xd, u, Nz, ps, Ny_r at the current time / state

    returns tuples if more than one aircraft
    '''

    llc = ap.llc
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = full_state.size // num_vars

    xd_tup = []
    u_tup = []
    Nz_tup = []
    ps_tup = []
    Ny_r_tup = []

    u_refs = ap.get_checked_u_ref(t, full_state)

    for i in range(num_aircraft):
        state = full_state[num_vars*i:num_vars*(i+1)]
        u_ref = u_refs[4*i:4*(i+1)]

        xd, u, Nz, ps, Ny_r = controlled_f16(t, state, u_ref, llc, model_str, v2_integrators)

        xd_tup.append(xd)
        u_tup.append(u)
        Nz_tup.append(Nz)
        ps_tup.append(ps)
        Ny_r_tup.append(Ny_r)

    if num_aircraft == 1:
        rv_xd = xd_tup[0]
        rv_u = u_tup[0]
        rv_Nz = Nz_tup[0]
        rv_ps = ps_tup[0]
        rv_Ny_r = Ny_r_tup[0]
    else:
        rv_xd = tuple(xd_tup)
        rv_u = tuple(u_tup)
        rv_Nz = tuple(Nz_tup)
        rv_ps = tuple(ps_tup)
        rv_Ny_r = tuple(Ny_r_tup)

    return rv_xd, rv_u, rv_Nz, rv_ps, rv_Ny_r


# TODO implement f16_env
import gym
from gym import spaces

#         x[0] = air speed, VT    (ft/sec)
#         x[1] = angle of attack, alpha  (rad)
#         x[2] = angle of sideslip, beta (rad)
#         x[3] = roll angle, phi  (rad)
#         x[4] = pitch angle, theta  (rad)
#         x[5] = yaw angle, psi  (rad)
#         x[6] = roll rate, P  (rad/sec)
#         x[7] = pitch rate, Q  (rad/sec)
#         x[8] = yaw rate, R  (rad/sec)
#         x[9] = northward horizontal displacement, pn  (feet)
#         x[10] = eastward horizontal displacement, pe  (feet)
#         x[11] = altitude, h  (feet)
#         x[12] = engine thrust dynamics lag state, pow
#
#         u[0] = throttle command  0.0 < u(1) < 1.0
#         u[1] = elevator command in degrees
#         u[2] = aileron command in degrees
#         u[3] = rudder command in degrees
#


class F16EnvV0(gym.Env):
    def __init__(self, args=None, pid=None):
        super().__init__()
        self.args = args
        self.pid = pid

        self.extended_states = True
        self.model_str='morelli'
        self.v2_integrators = False
        self.tmax = 3.51
        self.dt = 1./30

        x_limits = {
            "vt": [-10000., 10000.],
            "alpha": [-10000., 10000.],
            "beta": [-10000., 10000.],
            
            "phi": [-10000., 10000.],
            "theta": [-10000., 10000.],
            "psi": [-10000., 10000.],

            "P": [-10000., 10000.],
            "Q": [-10000., 10000.],
            "R": [-10000., 10000.],

            "pn": [-10000., 10000.],
            "pe": [-10000., 10000.],
            "h": [-10000., 10000.],

            "pow": [-10000., 10000.],

            "nz": [-10000., 10000.],
            "ps": [-10000., 10000.],
            "nyr": [-10000., 10000.],
        }

        u_limits = {
            "throttle": [-10000., 10000.],
            "nz": [-10000., 10000.],
            "ps": [-10000., 10000.],
            "nyr": [-10000., 10000.],
        }

        x_lows = np.array([x_limits[key][0] for key in x_limits], dtype=np.float32)
        x_highs = np.array([x_limits[key][1] for key in x_limits], dtype=np.float32)

        u_lows = np.array([u_limits[key][0] for key in u_limits], dtype=np.float32)
        u_highs = np.array([u_limits[key][1] for key in u_limits], dtype=np.float32)

        # TODO adding time variable?
        # TODO normalization & actuation?
        self.observation_space = spaces.Box(low=x_lows, high=x_highs, shape=(len(x_lows),))
        self.action_space = spaces.Box(low=u_lows, high=u_highs, shape=(len(u_lows),))

    def reset(self):
        self.t = 0
        self.ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')

        ### Initial Conditions ###
        power = 9 # engine power level (0-10)

        # Default alpha & beta
        alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
        beta = 0                # Side slip angle (rad)

        # Initial Attitude
        alt = 1000        # altitude (ft)
        vt = 540          # initial velocity (ft/sec)
        phi = -math.pi/8           # Roll angle from wings level (rad)
        theta = (-math.pi/2)*0.3         # Pitch angle from nose level (rad)
        psi = 0   # Yaw angle from North (rad)
        num_trials = 1

        power_sim = power
        alpha_min = alpha * 0.9
        alpha_max = alpha * 1.1
        alt_min = alt * 0.9
        alt_max = alt * 1.1
        vt_min = vt * 0.9
        vt_max = vt * 1.1
        phi_min = phi * 1.1
        phi_max = phi * 0.9
        theta_min = theta * 1.1
        theta_max = theta * 0.9
        psi_sim = psi

        beta_sim = beta

        alpha_sims = np.random.rand(num_trials) * (alpha_max-alpha_min) + alpha_min
        alt_sims = np.random.rand(num_trials) * (alt_max - alt_min) + alt_min
        vt_sims = np.random.rand(num_trials) * (vt_max - vt_min) + vt_min
        phi_sims = np.random.rand(num_trials) * (phi_max - phi_min) + phi_min
        theta_sims = np.random.rand(num_trials) * (theta_max - theta_min) + theta_min

        i=0
        init = [vt_sims[i], alpha_sims[i], beta_sim,
                phi_sims[i], theta_sims[i], psi_sim, 0, 0, 0, 0, 0, alt_sims[i], power_sim]

        initial_state = np.array(init, dtype=float)
        num_vars = len(get_state_names()) + self.ap.llc.get_num_integrators()
        if initial_state.size < num_vars:
            # append integral error states to state vector
            x0 = np.zeros(num_vars)
            x0[:initial_state.shape[0]] = initial_state
        else:
            x0 = initial_state

        # print(x0)
        # exit()

        self.times=[0]
        self.states=[x0]
        self.ap.advance_discrete_mode(self.times[-1], self.states[-1])

        self.modes = [self.ap.mode]

        if self.extended_states:
            xd, u, Nz, ps, Ny_r = get_extended_states(self.ap, self.times[-1], self.states[-1], self.model_str, self.v2_integrators)
            self.xd_list = [xd]
            self.u_list = [u]
            self.Nz_list = [Nz]
            self.ps_list = [ps]
            self.Ny_r_list = [Ny_r]
            # print(self.states[-1], u)

        # self.der_func = make_rl_func(self.ap, self.model_str, self.v2_integrators)

        # self.integrator = RK45(self.der_func, self.times[-1], self.states[-1], self.tmax)
        
        obs = self.get_observation()
        return obs

    def step(self, action):
        # TODO(to be completed)
        self.der_func = make_rl_func(action, self.ap, self.model_str, self.v2_integrators)
        self.integrator = RK45(self.der_func, self.times[-1], self.states[-1], self.times[-1]+self.dt)
        while self.integrator.t < self.times[-1]+self.dt:
            self.integrator.step()
        
        new_state = self.integrator.y

        self.times.append(t)
        self.states.append(new_state)
        updated = self.ap.advance_discrete_mode(self.times[-1], self.states[-1])
        self.modes.append(self.ap.mode)

        if self.extended_states:
            xd, u, Nz, ps, Ny_r = get_extended_states(self.ap, self.times[-1], self.states[-1], self.model_str, self.v2_integrators)
            # print(self.states[-1], u)
            self.xd_list.append(xd)
            self.u_list.append(u)
            self.Nz_list.append(Nz)
            self.ps_list.append(ps)
            self.Ny_r_list.append(Ny_r)
            # print("OURS",self.t, self.states[-1][11], "%.3f %.3f %.3f %.3f %.3f %.3f %.3f "%(u[0],u[1],u[2],u[3],u[4],u[5],u[6]))

        done = False

        if self.ap.is_finished(self.times[-1], self.states[-1]):
            self.integrator.status = 'autopilot finished'
            done = True

        if updated:
            self.integrator=RK45(self.der_func, self.times[-1], self.states[-1], self.times[-1]+self.dt)

        observation = self.get_observation()

        # other done conditions
        # TODO too large state exception
        done = done or (self.t > int(self.tmax//self.dt) or self.states[-1][11]<0)

        # rewards
        reward = self.get_reward()

        # info 
        info = {}

        self.t += 1

        return observation, reward, done, info

    def get_observation(self):
        observation = self.states[-1]
        return observation

    # TODO this needs further thoughts
    def get_reward(self):
        reward = 0
        if self.states[-1][11]<0:
            reward += min(0.0, self.states[-1][11]) * 10
        
        if self.states[-1][11]>0:
            reward += max(-self.states[-1][11], -10) * 0.1
        return reward
    
    def render(self, mode=None):
        return None
    

def make_rl_func(u_refs, ap, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'
        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft
        xds = []
        for i in range(num_aircraft):
            state = full_state[num_vars*i:num_vars*(i+1)]
            u_ref = u_refs[4*i:4*(i+1)]

            xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv
    
    return der_func


if __name__ == '__main__':
    t1=time.time()
    np.random.seed(1007)
    env = F16EnvV0()
    obs = env.reset()

    next_obs_list=[]
    t=0
    for i in range(int(3.51//(env.dt))):
        u_refs = env.ap.get_checked_u_ref(t, obs)
        next_obs, reward, done, _ = env.step(u_refs)
        obs = next_obs
        t+=env.dt
    t2=time.time()
    print("Elapsed %05f seconds"%(t2-t1))
