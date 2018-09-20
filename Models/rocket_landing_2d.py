import numpy as np
import cvxpy as cvx
import sympy as sp

from global_parameters import K


class Model:
    """
    A 2D path rocket landing problem.
    """
    n_x = 6
    n_u = 2

    m = 2
    I = 1e-2

    r_init = [4., 4.]
    v_init = [-2., -1.]
    t_init = [np.deg2rad(0.), ]
    w_init = [np.deg2rad(0.), ]

    r_final = [0., 0.]
    v_final = [0., 0.]
    t_final = [np.deg2rad(0.), ]
    w_final = [np.deg2rad(0.), ]

    t_f_guess = 10  # s

    t_max = np.deg2rad(60)
    w_max = np.deg2rad(60)
    max_gimbal = np.deg2rad(7)
    T_max = 5
    T_min = T_max * 0.4
    r_T = 1e-2
    g = 1

    def __init__(self):
        self.r_scale = np.linalg.norm(self.r_init)
        self.m_scale = self.m

        self.x_init = np.concatenate((self.r_init, self.v_init, self.t_init, self.w_init))
        self.x_final = np.concatenate((self.r_final, self.v_final, self.t_final, self.w_final))

    def nondimensionalize(self):
        """ nondimensionalize all parameters and boundaries """
        self.r_init /= self.r_scale
        self.v_init /= self.r_scale
        self.r_T /= self.r_scale
        self.g /= self.r_scale
        self.I /= self.m_scale * self.r_scale ** 2
        self.m /= self.m_scale
        self.T_min /= self.m_scale * self.r_scale
        self.T_max /= self.m_scale * self.r_scale

        self.x_init = self.x_nondim(self.x_init)
        self.x_final = self.x_nondim(self.x_final)

    def x_nondim(self, x):
        """ nondimensionalize a single x row """
        x[0:4] /= self.r_scale
        return x

    def u_nondim(self, u):
        """ nondimensionalize u"""
        u[1, :] /= self.m_scale * self.r_scale
        return u

    def redimensionalize(self):
        """ redimensionalize all parameters """
        self.r_init *= self.r_scale
        self.v_init *= self.r_scale
        self.r_T *= self.r_scale
        self.g *= self.r_scale
        self.I *= self.m_scale * self.r_scale ** 2
        self.m *= self.m_scale
        self.T_min *= self.m_scale * self.r_scale
        self.T_max *= self.m_scale * self.r_scale

        self.x_init = self.x_redim(self.x_init)
        self.x_final = self.x_redim(self.x_final)

    def x_redim(self, x):
        """ redimensionalize x, assumed to have the shape of a solution """
        x[0:4] *= self.r_scale
        return x

    def u_redim(self, u):
        """ redimensionalize u """
        u[1, :] *= self.m_scale * self.r_scale
        return u

    def get_equations(self):
        """
        :return: Functions to calculate A, B and f given state x and input u
        """
        f = sp.zeros(6, 1)

        x = sp.Matrix(sp.symbols('rx ry vx vy t w', real=True))
        u = sp.Matrix(sp.symbols('gimbal T', real=True))

        f[0, 0] = x[2, 0]
        f[1, 0] = x[3, 0]
        f[2, 0] = 1 / self.m * sp.sin(x[4, 0] + u[0, 0]) * u[1, 0]
        f[3, 0] = 1 / self.m * (sp.cos(x[4, 0] + u[0, 0]) * u[1, 0] - self.m * self.g)
        f[4, 0] = x[5, 0]
        f[5, 0] = 1 / self.I * (-sp.sin(u[0, 0]) * u[1, 0] * self.r_T)

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

        for k in range(K):
            alpha1 = (K - k) / K
            alpha2 = k / K

            X[:, k] = alpha1 * self.x_init + alpha2 * self.x_final
            U[0, :] = 0
            U[1, :] = (self.T_max - self.T_min) / 2

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
        objective = None
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

        constraints = [
            # Boundary conditions:
            X_v[0:2, 0] == self.x_init[0:2],
            X_v[2:4, 0] == self.x_init[2:4],
            X_v[4, 0] == self.x_init[4],
            X_v[5, 0] == self.x_init[5],

            X_v[:, -1] == self.x_final,

            # State constraints:
            cvx.abs(X_v[4, :]) <= self.t_max,
            cvx.abs(X_v[5, :]) <= self.w_max,
            X_v[1, :] >= 0,

            # Control constraints:
            cvx.abs(U_v[0, :]) <= self.max_gimbal,
            U_v[1, :] >= self.T_min,
            U_v[1, :] <= self.T_max,
        ]
        return constraints

    def get_linear_cost(self):
        return 0

    def get_nonlinear_cost(self, X, U=None):
        return 0
