import numpy as np
import cvxpy as cvx
import sympy as sp

from global_parameters import K


class Model:
    """
    A 2D path planning problem.
    """
    n_x = 3
    n_u = 2

    v_max = 1  # m/s
    w_max = np.pi / 6  # rad/s
    upper_bound = 10.  # m
    lower_bound = -10.  # m

    robot_radius = 0.5  # m

    r_init = np.array([-8., -8., 0.])
    r_final = np.array([8., 8., 0.])

    t_f_guess = 30  # s

    # slack variables for artificial infeasibility
    s_prime = []

    def __init__(self):
        self.x_init = self.r_init
        self.x_final = self.r_final

        self.r_scale = self.upper_bound - self.lower_bound

        # cylindrical obstacles [(x,y),r]
        self.obstacles = [
            [[5., 4.], 3.],
            [[-5., -4.], 3.],
            [[0., 0.], 2.],
        ]

        for _ in self.obstacles:
            self.s_prime.append(cvx.Variable((K, 1), nonneg=True))

    def nondimensionalize(self):
        """ nondimensionalize all parameters and boundaries """

        self.v_max /= self.r_scale
        self.upper_bound /= self.r_scale
        self.lower_bound /= self.r_scale
        self.robot_radius /= self.r_scale

        self.x_init = self.x_nondim(self.x_init)
        self.x_final = self.x_nondim(self.x_final)

        for j in range(len(self.obstacles)):
            self.obstacles[j][0][0] /= self.r_scale
            self.obstacles[j][0][1] /= self.r_scale
            self.obstacles[j][1] /= self.r_scale

    def x_nondim(self, x):
        """ nondimensionalize a single x row """
        x[0:2] /= self.r_scale
        return x

    def u_nondim(self, u):
        """ nondimensionalize u"""
        u[0, :] /= self.r_scale
        return u

    def redimensionalize(self):
        """ redimensionalize all parameters """
        self.v_max *= self.r_scale
        self.upper_bound *= self.r_scale
        self.lower_bound *= self.r_scale
        self.robot_radius *= self.r_scale

        self.x_init = self.x_nondim(self.x_init)
        self.x_final = self.x_nondim(self.x_final)

        for j in range(len(self.obstacles)):
            self.obstacles[j][0][0] *= self.r_scale
            self.obstacles[j][0][1] *= self.r_scale
            self.obstacles[j][1] *= self.r_scale

    def x_redim(self, x):
        """ redimensionalize x, assumed to have the shape of a solution """
        x[0:2, :] *= self.r_scale
        return x

    def u_redim(self, u):
        """ redimensionalize u """
        u[0, :] *= self.r_scale
        return u

    def get_equations(self):
        """
        :return: Functions to calculate A, B and f given state x and input u
        """
        f = sp.zeros(3, 1)

        x = sp.Matrix(sp.symbols('x y theta', real=True))
        u = sp.Matrix(sp.symbols('v w', real=True))

        f[0, 0] = u[0, 0] * sp.cos(x[2, 0])
        f[1, 0] = u[0, 0] * sp.sin(x[2, 0])
        f[2, 0] = u[1, 0]

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
            X[:, k] = self.x_init * alpha1 + self.x_final * alpha2

        U[:, :] = 0

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

        slack = 0
        for j in range(len(self.obstacles)):
            slack += cvx.sum(self.s_prime[j])

        objective = cvx.Minimize(1e5 * slack)
        # objective += cvx.Minimize(cvx.sum(cvx.square(U_v)))
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
            X_v[:, 0] == self.x_init,
            X_v[:, -1] == self.x_final,

            U_v[:, 0] == 0,
            U_v[:, -1] == 0
        ]

        # Input conditions:
        constraints += [
            0 <= U_v[0, :],
            U_v[0, :] <= self.v_max,
            cvx.abs(U_v[1, :]) <= self.w_max,
        ]

        # State conditions:
        constraints += [
            X_v[0:2, :] <= self.upper_bound - self.robot_radius,
            X_v[0:2, :] >= self.lower_bound + self.robot_radius,
        ]

        # linearized obstacles
        for j, obst in enumerate(self.obstacles):
            p = obst[0]
            r = obst[1] + self.robot_radius

            lhs = [(X_last_p[0:2, k] - p) / (cvx.norm((X_last_p[0:2, k] - p)) + 1e-6) * (X_v[0:2, k] - p)
                   for k in range(K)]
            constraints += [r - cvx.vstack(lhs) <= self.s_prime[j]]
        return constraints

    def get_linear_cost(self):
        cost = 0
        for j in range(len(self.obstacles)):
            cost += np.sum(self.s_prime[j].value)
        return cost

    def get_nonlinear_cost(self, X, U=None):
        cost = 0
        for obst in self.obstacles:
            vector_to_obstacle = X[0:2, :].T - obst[0]
            dist_to_obstacle = np.linalg.norm(vector_to_obstacle, 2, axis=1)
            is_violated = dist_to_obstacle < obst[1] + self.robot_radius
            violation = obst[1] + self.robot_radius - dist_to_obstacle
            cost += np.sum(is_violated * violation)
        return cost
