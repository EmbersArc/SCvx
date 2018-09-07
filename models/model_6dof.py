import sympy as sp
import numpy as np
import cvxpy as cvx
from utils import euler_to_quat


def skew(v):
    return sp.Matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def dir_cosine(q):
    return sp.Matrix([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
        [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
        [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
    ])


def omega(w):
    return sp.Matrix([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0],
    ])


class Model_6DoF:
    """
    A 6 degree of freedom rocket landing problem.
    """
    n_x = 14
    n_u = 3

    # Mass
    m_wet = 30000.  # 30000 kg
    m_dry = 22000.  # 22000 kg

    # Flight time guess
    t_f_guess = 15.  # 10 s

    # State constraints
    r_I_init = np.array((200., 200., 0.))  # 2000 m, 200 m, 200 m
    v_I_init = np.array((-50., -100., -50.))  # -300 m/s, 50 m/s, 50 m/s
    q_B_I_init = euler_to_quat((0, 0, 0))
    w_B_init = np.deg2rad(np.array((0., 0., 0.)))

    r_I_final = np.array((0., 0., 0.))
    v_I_final = np.array((-10, 0., 0.))
    q_B_I_final = euler_to_quat((0, 0, 0))
    w_B_final = np.deg2rad(np.array((0., 0., 0.)))

    w_B_max = np.deg2rad(90)

    # Angles
    max_gimbal = 7
    max_angle = 90
    glidelslope_angle = 20

    tan_delta_max = np.tan(np.deg2rad(max_gimbal))
    cos_theta_max = np.cos(np.deg2rad(max_angle))
    tan_gamma_gs = np.tan(np.deg2rad(glidelslope_angle))

    # Thrust limits
    T_max = 800000.  # 800000 [kg*m/s^2]
    T_min = T_max * 0.3

    # Angular moment of inertia
    J_B = np.diag([100000., 4000000., 4000000.])  # 100000 [kg*m^2], 4000000 [kg*m^2], 4000000 [kg*m^2]

    # Gravity
    g_I = np.array((-9.81, 0., 0.))  # -9.81 [m/s^2]

    # Fuel consumption
    alpha_m = 1 / (282 * 9.81)  # 1 / (282 * 9.81) [s/m]

    # Vector from thrust point to CoM
    r_T_B = np.array([-14, 0., 0.])  # -20 m

    def set_random_initial_state(self):
        self.r_I_init[0] = np.random.uniform(350, 400)
        self.r_I_init[1:3] = np.random.uniform(-200, 200, size=2)

        self.v_I_init[0] = np.random.uniform(-100, -80)
        self.v_I_init[1:3] = np.random.uniform(-0.5, -0.2, size=2) * self.r_I_init[1:3]

        self.q_B_I_init = euler_to_quat((0,
                                         np.random.uniform(-30, 30),
                                         np.random.uniform(-30, 30)))
        self.w_B_init = np.deg2rad((0,
                                    np.random.uniform(-20, 20),
                                    np.random.uniform(-20, 20)))

    # ------------------------------------------ Start normalization stuff
    def __init__(self):
        """
        A large r_scale for a small scale problem will
        ead to numerical problems as parameters become excessively small
        and (it seems) precision is lost in the dynamics.
        """

        self.set_random_initial_state()

        self.x_init = np.concatenate(((self.m_wet,), self.r_I_init, self.v_I_init, self.q_B_I_init, self.w_B_init))
        self.x_final = np.concatenate(((self.m_dry,), self.r_I_final, self.v_I_final, self.q_B_I_final, self.w_B_final))

        self.r_scale = np.linalg.norm(self.r_I_init)
        self.m_scale = self.m_wet

        self.nondimensionalize()

    def nondimensionalize(self):
        """ nondimensionalize all parameters and boundaries """

        self.alpha_m *= self.r_scale  # s
        self.r_T_B /= self.r_scale  # 1
        self.g_I /= self.r_scale  # 1/s^2
        self.J_B /= (self.m_scale * self.r_scale ** 2)  # 1

        self.x_init = self.x_nondim(self.x_init)
        self.x_final = self.x_nondim(self.x_final)

        self.T_max = self.u_nondim(self.T_max)
        self.T_min = self.u_nondim(self.T_min)

        self.m_wet /= self.m_scale
        self.m_dry /= self.m_scale

    def x_nondim(self, x):
        """ nondimensionalize a single x row """

        x[0] /= self.m_scale
        x[1:4] /= self.r_scale
        x[4:7] /= self.r_scale

        return x

    def u_nondim(self, u):
        """ nondimensionalize u, or in general any force in Newtons"""
        return u / (self.m_scale * self.r_scale)

    def redimensionalize(self):
        """ redimensionalize all parameters """

        self.alpha_m /= self.r_scale  # s
        self.r_T_B *= self.r_scale
        self.g_I *= self.r_scale
        self.J_B *= (self.m_scale * self.r_scale ** 2)

        self.T_max = self.u_redim(self.T_max)
        self.T_min = self.u_redim(self.T_min)

        self.m_wet *= self.m_scale
        self.m_dry *= self.m_scale

    def x_redim(self, x):
        """ redimensionalize x, assumed to have the shape of a solution """

        x[0, :] *= self.m_scale
        x[1:4, :] *= self.r_scale
        x[4:7, :] *= self.r_scale

        return x

    def u_redim(self, u):
        """ redimensionalize u """
        return u * (self.m_scale * self.r_scale)

    # ------------------------------------------ End normalization stuff

    def get_equations(self):
        """
        :return: Functions to calculate A, B and f given state x and input u
        """
        f = sp.zeros(14, 1)

        x = sp.Matrix(sp.symbols('m rx ry rz vx vy vz q0 q1 q2 q3 wx wy wz', real=True))
        u = sp.Matrix(sp.symbols('ux uy uz', real=True))

        g_I = sp.Matrix(self.g_I)
        r_T_B = sp.Matrix(self.r_T_B)
        J_B = sp.Matrix(self.J_B)

        C_B_I = dir_cosine(x[7:11, 0])
        C_I_B = C_B_I.transpose()

        f[0, 0] = - self.alpha_m * u.norm()
        f[1:4, 0] = x[4:7, 0]
        f[4:7, 0] = 1 / x[0, 0] * C_I_B * u + g_I
        f[7:11, 0] = 1 / 2 * omega(x[11:14, 0]) * x[7: 11, 0]
        f[11:14, 0] = J_B ** -1 * (skew(r_T_B) * u - skew(x[11:14, 0]) * J_B * x[11:14, 0])

        f = sp.simplify(f)
        A = sp.simplify(f.jacobian(x))
        B = sp.simplify(f.jacobian(u))

        f_func = sp.lambdify((x, u), f, 'numpy')
        A_func = sp.lambdify((x, u), A, 'numpy')
        B_func = sp.lambdify((x, u), B, 'numpy')

        return f_func, A_func, B_func

    def initialize_trajectory(self, X, U):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """
        K = X.shape[1]

        for k in range(K):
            alpha1 = (K - k) / K
            alpha2 = k / K

            m_k = (alpha1 * self.x_init[0] + alpha2 * self.x_final[0],)
            r_I_k = alpha1 * self.x_init[1:4] + alpha2 * self.x_final[1:4]
            v_I_k = alpha1 * self.x_init[4:7] + alpha2 * self.x_final[4:7]
            q_B_I_k = np.array([1, 0, 0, 0])
            w_B_k = alpha1 * self.x_init[11:14] + alpha2 * self.x_final[11:14]

            X[:, k] = np.concatenate((m_k, r_I_k, v_I_k, q_B_I_k, w_B_k))
            U[:, k] = m_k * -self.g_I

        return X, U

    def get_objective(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific objective to be minimized.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A cvx objective function.
        """
        # objective = cvx.Minimize(100 * cvx.norm(X_v[2:4, -1]))
        objective = cvx.Minimize(cvx.sum(cvx.norm(U_v, axis=0)))
        return objective

    def get_constraints(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific constraints.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A list of cvx constraints
        """
        # Boundary conditions:
        constraints = [
            X_v[0, 0] == self.x_init[0],
            X_v[1:4, 0] == self.x_init[1:4],
            X_v[4:7, 0] == self.x_init[4:7],
            X_v[7:11, 0] == self.x_init[7:11],
            X_v[11:14, 0] == self.x_init[11:14],

            # X_[0, -1] == self.x_final[0], # final mass is free
            X_v[1:, -1] == self.x_final[1:],
            U_v[1:3, -1] == 0,
        ]

        # smooth control input
        # constraints += [cvx.norm(U_v[0:-1, :] - U_v[1:, :], axis=0) <= 0.1]

        constraints += [
            # State constraints:
            X_v[0, :] >= self.m_dry,  # minimum mass
            cvx.norm(X_v[2: 4, :], axis=0) <= X_v[1, :] / self.tan_gamma_gs,  # glideslope
            cvx.norm(X_v[9:11, :], axis=0) <= np.sqrt((1 - self.cos_theta_max) / 2),  # maximum angle
            cvx.norm(X_v[11: 14, :], axis=0) <= self.w_B_max,  # maximum angular velocity

            # Control constraints:
            cvx.norm(U_v[1:3, :], axis=0) <= self.tan_delta_max * U_v[0, :],  # gimbal angle constraint
            cvx.norm(U_v, axis=0) <= self.T_max,  # upper thrust constraint
            U_v[0, :] >= self.T_min  # simple lower thrust constraint
        ]

        # # linearized lower thrust constraint
        # rhs = [U_last_p[:, k] / cvx.norm(U_last_p[:, k]) * U_v[:, k] for k in range(X_v.shape[1])]
        # constraints += [
        #     self.T_min <= cvx.vstack(rhs)
        # ]

        return constraints
