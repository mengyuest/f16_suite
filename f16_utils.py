'''gcas autopilot

copied from matlab v2
'''

import math

import numpy as np
from numpy import deg2rad

#TODO util.py
'''
Utilities for F-16 GCAS
'''

from math import floor, ceil
import numpy as np

class StateIndex:
    'list of static state indices'

    VT = 0
    VEL = 0 # alias
    
    ALPHA = 1
    BETA = 2
    PHI = 3 # roll angle
    THETA = 4 # pitch angle
    PSI = 5 # yaw angle
    
    P = 6
    Q = 7
    R = 8
    
    POSN = 9
    POS_N = 9
    
    POSE = 10
    POS_E = 10
    
    ALT = 11
    H = 11
    
    POW = 12

class Freezable():
    'a class where you can freeze the fields (prevent new fields from being created)'

    _frozen = False

    def freeze_attrs(self):
        'prevents any new attributes from being created in the object'
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError("{} does not contain attribute '{}' (object was frozen)".format(self, key))

        object.__setattr__(self, key, value)

class Euler(Freezable):
    '''fixed step euler integration

    loosely based on scipy.integrate.RK45
    '''

    def __init__(self, der_func, tstart, ystart, tend, step=0, time_tol=1e-9):
        assert step > 0, "arg step > 0 required in Euler integrator"
        assert tend > tstart

        self.der_func = der_func # signature (t, x)
        self.tstep = step
        self.t = tstart
        self.y = ystart.copy()
        self.yprev = None
        self.tprev = None
        self.tend = tend

        self.status = 'running'

        self.time_tol = time_tol

        self.freeze_attrs()

    def step(self):
        'take one step'

        if self.status == 'running':
            self.yprev = self.y.copy()
            self.tprev = self.t
            yd = self.der_func(self.t, self.y)

            self.t += self.tstep

            if self.t + self.time_tol >= self.tend:
                self.t = self.tend

            dt = self.t - self.tprev
            self.y += dt * yd

            if self.t == self.tend:
                self.status = 'finished'

    def dense_output(self):
        'return a function of time'

        assert self.tprev is not None

        dy = self.y - self.yprev
        dt = self.t - self.tprev

        dydt = dy / dt

        def fun(t):
            'return state at time t (linear interpolation)'

            deltat = t - self.tprev

            return self.yprev + dydt * deltat

        return fun

def get_state_names():
    'returns a list of state variable names'

    return ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'P', 'Q', 'R', 'pos_n', 'pos_e', 'alt', 'pow']

def printmat(mat, main_label, row_label_str, col_label_str):
    'print a matrix'

    if isinstance(row_label_str, list) and len(row_label_str) == 0:
        row_label_str = None

    assert isinstance(main_label, str)
    assert row_label_str is None or isinstance(row_label_str, str)
    assert isinstance(col_label_str, str)

    mat = np.array(mat)
    if len(mat.shape) == 1:
        mat.shape = (1, mat.shape[0]) # one-row matrix

    print("{main_label} =")

    row_labels = None if row_label_str is None else row_label_str.split(' ')
    col_labels = col_label_str.split(' ')

    width = 7

    width = max(width, max([len(l) for l in col_labels]))

    if row_labels is not None:
        width = max(width, max([len(l) for l in row_labels]))

    width += 1

    # add blank space for row labels
    if row_labels is not None:
        print("{: <{}}".format('', width), end='')

    # print col lables
    for col_label in col_labels:
        if len(col_label) > width:
            col_label = col_label[:width]

        print("{: >{}}".format(col_label, width), end='')

    print('')

    if row_labels is not None:
        assert len(row_labels) == mat.shape[0], \
            "row labels (len={}) expected one element for each row of the matrix ({})".format( \
            len(row_labels), mat.shape[0])

    for r in range(mat.shape[0]):
        row = mat[r]

        if row_labels is not None:
            label = row_labels[r]

            if len(label) > width:
                label = label[:width]

            print("{:<{}}".format(label, width), end='')

        for num in row:
            #print("{:#<{}}".format(num, width), end='')
            print("{:{}.{}g}".format(num, width, width-3), end='')

        print('')


from f16_sub_utils import fix, sign

def extract_single_result(res, index, llc):
    'extract a res object for a sinlge aircraft from a multi-aircraft simulation'

    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = res['states'][0].size // num_vars

    if num_aircraft == 1:
        assert index == 0
        rv = res
    else:
        rv = {}
        rv['status'] = res['status']
        rv['times'] = res['times']
        rv['modes'] = res['modes']

        full_states = res['states']
        rv['states'] = full_states[:, num_vars*index:num_vars*(index+1)]

        if 'xd_list' in res:
            # extended states
            key_list = ['xd_list', 'ps_list', 'Nz_list', 'Ny_r_list', 'u_list']

            for key in key_list:
                rv[key] = [tup[index] for tup in res[key]]

    return rv

class SafetyLimits(Freezable):
    'a class for holding a set of safety limits.'

    def __init__(self, **kwargs):
        self.altitude = kwargs['altitude'] if 'altitude' in kwargs and kwargs['altitude'] is not None else None
        self.Nz = kwargs['Nz'] if 'Nz' in kwargs and kwargs['Nz'] is not None else None
        self.v = kwargs['v'] if 'v' in kwargs and kwargs['v'] is not None else None
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs and kwargs['alpha'] is not None else None

        self.psMaxAccelDeg = kwargs['psMaxAccelDeg'] if 'psMaxAccelDeg' in kwargs and kwargs['psMaxAccelDeg'] is not None else None
        self.betaMaxDeg = kwargs['betaMaxDeg'] if 'betaMaxDeg' in kwargs and kwargs['betaMaxDeg'] is not None else None

        self.freeze_attrs()

class SafetyLimitsVerifier(Freezable):
    'given some limits (in a SafetyLimits) and optional low-level controller (in a LowLevelController), verify whether the simulation results are safe.'

    def __init__(self, safety_limits, llc=None):
        self.safety_limits = safety_limits
        self.llc = llc

    def verify(self, results):
        # Determine the number of state variables per tick of the simulation.
        if self.llc is not None:
            num_state_vars = len(get_state_names()) + \
                             self.llc.get_num_integrators()
        else:
            num_state_vars = len(get_state_names())
        # Check whether the results are sane.
        assert (results['states'].size % num_state_vars) == 0, \
            "Wrong number of state variables."

        # Go through each tick of the simulation and determine whether
        # the object(s) was (were) in a safe state.
        for i in range(results['states'].size // num_state_vars):
            _vt, alpha, beta, _phi, \
                _theta, _psi, _p, _q, \
                _r, _pos_n, _pos_e, alt, \
                _, _, _, _ = results['states'][i]
            nz = results['Nz_list'][i]
            ps = results['ps_list'][i]

            if self.safety_limits.altitude is not None:
                assert self.safety_limits.altitude[0] <= alt <= self.safety_limits.altitude[1], "Altitude ({}) is not within the specified limits ({}, {}).".format(alt, self.safety_limits.altitude[0], self.safety_limits.altitude[1])

            if self.safety_limits.Nz is not None:
                assert self.safety_limits.Nz[0] <= nz <= self.safety_limits.Nz[1], "Nz ({}) is not within the specified limits ({}, {}).".format(nz, self.safety_limits.Nz[0], self.safety_limits.Nz[1])

            if self.safety_limits.alpha is not None:
                assert self.safety_limits.alpha[0] <= alpha <= self.safety_limits.alpha[1], "alpha ({}) is not within the specified limits ({}, {}).".format(nz, self.safety_limits.alpha[0], self.safety_limits.alpha[1])

            if self.safety_limits.psMaxAccelDeg is not None:
                assert ps <= self.safety_limits.psMaxAccelDeg, "Ps is not less than the specified max."

            if self.safety_limits.betaMaxDeg is not None:
                assert beta <= self.safety_limits.betaMaxDeg, "Beta is not less than the specified max."



#TODO lowlevel/low_level_controller.py
class CtrlLimits(Freezable):
    'Control Limits'

    def __init__(self):
        self.ThrottleMax = 1 # Afterburner on for throttle > 0.7
        self.ThrottleMin = 0
        self.ElevatorMaxDeg = 25
        self.ElevatorMinDeg = -25
        self.AileronMaxDeg = 21.5
        self.AileronMinDeg = -21.5
        self.RudderMaxDeg = 30
        self.RudderMinDeg = -30
        
        self.NzMax = 6
        self.NzMin = -1

        self.freeze_attrs()

class LowLevelController(Freezable):
    '''low level flight controller
    '''

    old_k_long = np.array([[-156.8801506723475, -31.037008068526642, -38.72983346216317]], dtype=float)
    old_k_lat = np.array([[37.84483, -25.40956, -6.82876, -332.88343, -17.15997],
                          [-23.91233, 5.69968, -21.63431, 64.49490, -88.36203]], dtype=float)

    old_xequil = np.array([502.0, 0.0389, 0.0, 0.0, 0.0389, 0.0, 0.0, 0.0, \
                        0.0, 0.0, 0.0, 1000.0, 9.0567], dtype=float).transpose()
    old_uequil = np.array([0.1395, -0.7496, 0.0, 0.0], dtype=float).transpose()

    def __init__(self, gain_str='old'):
        # Hard coded LQR gain matrix from matlab version

        assert gain_str == 'old'

        # Longitudinal Gains
        K_long = LowLevelController.old_k_long
        K_lat = LowLevelController.old_k_lat

        self.K_lqr = np.zeros((3, 8))
        self.K_lqr[:1, :3] = K_long
        self.K_lqr[1:, 3:] = K_lat

        # equilibrium points from BuildLqrControllers.py
        self.xequil = LowLevelController.old_xequil
        self.uequil = LowLevelController.old_uequil

        self.ctrlLimits = CtrlLimits()

        self.freeze_attrs()

    def get_u_deg(self, u_ref, f16_state):
        'get the reference commands for the control surfaces'

        # Calculate perturbation from trim state
        x_delta = f16_state.copy()
        x_delta[:len(self.xequil)] -= self.xequil

        ## Implement LQR Feedback Control
        # Reorder states to match controller:
        # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
        x_ctrl = np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=float)

        # Initialize control vectors
        u_deg = np.zeros((4,)) # throt, ele, ail, rud

        # Calculate control using LQR gains
        u_deg[1:4] = np.dot(-self.K_lqr, x_ctrl) # Full Control

        # Set throttle as directed from output of getOuterLoopCtrl(...)
        u_deg[0] = u_ref[3]

        # Add in equilibrium control
        u_deg[0:4] += self.uequil

        ## Limit controls to saturation limits
        ctrlLimits = self.ctrlLimits

        # Limit throttle from 0 to 1
        u_deg[0] = max(min(u_deg[0], ctrlLimits.ThrottleMax), ctrlLimits.ThrottleMin)

        # Limit elevator from -25 to 25 deg
        u_deg[1] = max(min(u_deg[1], ctrlLimits.ElevatorMaxDeg), ctrlLimits.ElevatorMinDeg)

        # Limit aileron from -21.5 to 21.5 deg
        u_deg[2] = max(min(u_deg[2], ctrlLimits.AileronMaxDeg), ctrlLimits.AileronMinDeg)

        # Limit rudder from -30 to 30 deg
        u_deg[3] = max(min(u_deg[3], ctrlLimits.RudderMaxDeg), ctrlLimits.RudderMinDeg)

        return x_ctrl, u_deg

    def get_num_integrators(self):
        'get the number of integrators in the low-level controller'

        return 3

    def get_integrator_derivatives(self, t, x_f16, u_ref, Nz, ps, Ny_r):
        'get the derivatives of the integrators in the low-level controller'

        return [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]





#TODO highlevel/autopilot.py
'''
Stanley Bak
Autopilot State-Machine Logic

There is a high-level advance_discrete_state() function, which checks if we should change the current discrete state,
and a get_u_ref(f16_state) function, which gets the reference inputs at the current discrete state.
'''

import abc
from math import pi

import numpy as np
from numpy import deg2rad

class Autopilot(Freezable):
    '''A container object for the hybrid automaton logic for a particular autopilot instance'''

    def __init__(self, init_mode, llc=None):

        assert isinstance(init_mode, str), 'init_mode should be a string'

        if llc is None:
            # use default
            llc = LowLevelController()

        self.llc = llc
        self.xequil = llc.xequil
        self.uequil = llc.uequil
        
        self.mode = init_mode # discrete state, this should be overwritten by subclasses

        self.freeze_attrs()

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete mode based on the current aircraft state. Returns True iff the discrete mode
        has changed. It's also suggested to update self.mode to the current mode name.
        '''

        return False

    def is_finished(self, t, x_f16):
        '''
        returns True if the simulation should stop (for example, after maneuver completes)

        this is called after advance_discrete_state
        '''

        return False

    @abc.abstractmethod
    def get_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals. Override this one
        in subclasses.

        returns four values per aircraft: Nz, ps, Ny_r, throttle
        '''

        return

    def get_checked_u_ref(self, t, x_f16):
        '''
        for the current discrete state, get the reference inputs signals and check them against ctrl limits
        '''

        rv = np.array(self.get_u_ref(t, x_f16), dtype=float)

        assert rv.size % 4 == 0, "get_u_ref should return Nz, ps, Ny_r, throttle for each aircraft"

        for i in range(rv.size //4):
            Nz, _ps, _Ny_r, _throttle = rv[4*i:4*(i+1)]

            l, u = self.llc.ctrlLimits.NzMin, self.llc.ctrlLimits.NzMax
            assert l <= Nz <= u, f"autopilot commanded invalid Nz ({Nz}). Not in range [{l}, {u}]"

        return rv

class FixedSpeedAutopilot(Autopilot):
    '''Simple Autopilot that gives a fixed speed command using proportional control'''

    def __init__(self, setpoint, p_gain):
        self.setpoint = setpoint
        self.p_gain = p_gain

        init_mode = 'tracking speed'
        Autopilot.__init__(self, init_mode)

    def get_u_ref(self, t, x_f16):
        '''for the current discrete state, get the reference inputs signals'''

        x_dif = self.setpoint - x_f16[0]

        return 0, 0, 0, self.p_gain * x_dif


#TODO examples/gcas/gcas_autopilot.py
class GcasAutopilot(Autopilot):
    '''ground collision avoidance autopilot'''

    def __init__(self, init_mode='standby', gain_str='old', stdout=False):

        assert init_mode in ['standby', 'roll', 'pull', 'waiting']

        # config
        self.cfg_eps_phi = deg2rad(5)       # Max abs roll angle before pull
        self.cfg_eps_p = deg2rad(10)        # Max abs roll rate before pull
        self.cfg_path_goal = deg2rad(0)     # Min path angle before completion
        self.cfg_k_prop = 4                 # Proportional control gain
        self.cfg_k_der = 2                  # Derivative control gain
        self.cfg_flight_deck = 1000         # Altitude at which GCAS activates
        self.cfg_min_pull_time = 2          # Min duration of pull up

        self.cfg_nz_des = 5

        self.pull_start_time = 0
        self.stdout = stdout

        self.waiting_cmd = np.zeros(4)
        self.waiting_time = 2

        llc = LowLevelController(gain_str=gain_str)

        Autopilot.__init__(self, init_mode, llc=llc)

    def log(self, s):
        'print to terminal if stdout is true'

        if self.stdout:
            print(s)

    def advance_discrete_mode(self, t, x_f16):
        '''
        advance the discrete state based on the current aircraft state. Returns True iff the discrete state
        has changed.
        '''

        premode = self.mode

        if self.mode == 'waiting':
            # time-triggered start after two seconds
            if t + 1e-6 >= self.waiting_time:
                self.mode = 'roll'
        elif self.mode == 'standby':
            if not self.is_nose_high_enough(x_f16) and not self.is_above_flight_deck(x_f16):
                self.mode = 'roll'
        elif self.mode == 'roll':
            if self.is_roll_rate_low(x_f16) and self.are_wings_level(x_f16):
                self.mode = 'pull'
                self.pull_start_time = t
        else:
            assert self.mode == 'pull', f"unknown mode: {self.mode}"

            if self.is_nose_high_enough(x_f16) and t >= self.pull_start_time + self.cfg_min_pull_time:
                self.mode = 'standby'

        rv = premode != self.mode

        if rv:
            self.log(f"GCAS transition {premode} -> {self.mode} at time {t}")

        return rv

    def are_wings_level(self, x_f16):
        'are the wings level?'

        phi = x_f16[StateIndex.PHI]

        radsFromWingsLevel = round(phi / (2 * math.pi))

        return abs(phi - (2 * math.pi)  * radsFromWingsLevel) < self.cfg_eps_phi

    def is_roll_rate_low(self, x_f16):
        'is the roll rate low enough to switch to pull?'
        p = x_f16[StateIndex.P]

        return abs(p) < self.cfg_eps_p

    def is_above_flight_deck(self, x_f16):
        'is the aircraft above the flight deck?'

        alt = x_f16[StateIndex.ALT]

        return alt >= self.cfg_flight_deck

    def is_nose_high_enough(self, x_f16):
        'is the nose high enough?'

        theta = x_f16[StateIndex.THETA]
        alpha = x_f16[StateIndex.ALPHA]

        # Determine which angle is "level" (0, 360, 720, etc)
        radsFromNoseLevel = round((theta-alpha)/(2 * math.pi))

        # Evaluate boolean
        return (theta-alpha) - 2 * math.pi * radsFromNoseLevel > self.cfg_path_goal

    def get_u_ref(self, _t, x_f16):
        '''get the reference input signals'''

        if self.mode == 'standby':
            rv = np.zeros(4)
        elif self.mode == 'waiting':
            rv = self.waiting_cmd
        elif self.mode == 'roll':
            rv = self.roll_wings_level(x_f16)
        else:
            assert self.mode == 'pull', f"unknown mode: {self.mode}"
            rv = self.pull_nose_level()

        return rv

    def pull_nose_level(self):
        'get commands in mode PULL'

        rv = np.zeros(4)

        rv[0] = self.cfg_nz_des

        return rv

    def roll_wings_level(self, x_f16):
        'get commands in mode ROLL'

        phi = x_f16[StateIndex.PHI]
        p = x_f16[StateIndex.P]

        rv = np.zeros(4)

        # Determine which angle is "level" (0, 360, 720, etc)
        radsFromWingsLevel = round(phi / (2 * math.pi))

        # PD Control until phi == pi * radsFromWingsLevel
        ps = -(phi - (2 * math.pi) * radsFromWingsLevel) * self.cfg_k_prop - p * self.cfg_k_der

        # Build commands to roll wings level
        rv[1] = ps

        return rv
