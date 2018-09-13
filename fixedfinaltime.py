from time import time
import numpy as np

from FixedFinalTime.parameters import *
from FixedFinalTime.discretization import FirstOrderHold
from FixedFinalTime.scproblem import SCProblem
from utils import format_line, save_arrays

# from Models.diffdrive_2d import Model
# from Models.diffdrive_2d_plot import plot
from Models.rocket_landing_3d import Model
from Models.rocket_landing_3d_plot import plot

"""
Python implementation of the Successive Convexification algorithm.

Rocket trajectory model based on the
'Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time' paper
by Michael Szmuk and Behçet Açıkmeşe.

Implementation by Sven Niederberger (s-niederberger@outlook.com)
"""

m = Model()
m.nondimensionalize()

# state and input
X = np.empty(shape=[m.n_x, K])
U = np.empty(shape=[m.n_u, K])

# INITIALIZATION--------------------------------------------------------------------------------------------------------
sigma = m.t_f_guess
X, U = m.initialize_trajectory(X, U)

# START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
all_X = [m.x_redim(X.copy())]
all_U = [m.u_redim(U.copy())]

integrator = FirstOrderHold(m, K, sigma)
problem = SCProblem(m, K)

last_nonlinear_cost = None
converged = False
for it in range(iterations):
    t0_it = time()
    print('-' * 50)
    print('-' * 18 + f' Iteration {str(it + 1).zfill(2)} ' + '-' * 18)
    print('-' * 50)

    t0_tm = time()
    A_bar, B_bar, C_bar, z_bar = integrator.calculate_discretization(X, U)
    print(format_line('Time for transition matrices', time() - t0_tm, 's'))

    problem.set_parameters(A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, z_bar=z_bar,
                           X_last=X, U_last=U,
                           weight_nu=w_nu, tr_radius=tr_radius)

    while True:
        error = problem.solve(verbose=verbose_solver, solver=solver, max_iters=200)
        print(format_line('Solver Error', error))

        # get solution
        new_X = problem.get_variable('X')
        new_U = problem.get_variable('U')

        X_nl = integrator.integrate_nonlinear_piecewise(new_X, new_U)

        linear_cost_dynamics = np.linalg.norm(problem.get_variable('nu'), 1)
        nonlinear_cost_dynamics = np.linalg.norm(new_X - X_nl, 1)

        linear_cost_constraints = m.get_linear_cost()
        nonlinear_cost_constraints = m.get_nonlinear_cost(X=new_X, U=new_U)

        linear_cost = linear_cost_dynamics + linear_cost_constraints  # J
        nonlinear_cost = nonlinear_cost_dynamics + nonlinear_cost_constraints  # L

        if last_nonlinear_cost is None:
            last_nonlinear_cost = nonlinear_cost
            X = new_X
            U = new_U
            break

        actual_change = last_nonlinear_cost - nonlinear_cost  # delta_J
        predicted_change = last_nonlinear_cost - linear_cost  # delta_L

        print('')
        print(format_line('Virtual Control Cost', linear_cost_dynamics))
        print(format_line('Constraint Cost', linear_cost_constraints))
        print('')
        print(format_line('Actual change', actual_change))
        print(format_line('Predicted change', predicted_change))
        print('')

        if abs(predicted_change) < 1e-6:
            converged = True
            break
        else:
            rho = actual_change / predicted_change
            if rho < rho_0:
                # reject solution
                tr_radius /= alpha
                print(f'Trust region too large. Solving again with radius={tr_radius}')
            else:
                # accept solution
                X = new_X
                U = new_U

                print('Solution accepted.')

                if rho < rho_1:
                    print('Decreasing radius.')
                    tr_radius /= alpha
                elif rho >= rho_2:
                    print('Increasing radius.')
                    tr_radius *= beta

                last_nonlinear_cost = nonlinear_cost
                break

        problem.set_parameters(tr_radius=tr_radius)

        print('-' * 50)

    print('')
    print(format_line('Time for iteration', time() - t0_it, 's'))
    print('')

    all_X.append(m.x_redim(X.copy()))
    all_U.append(m.u_redim(U.copy()))

    if converged:
        print(f'Converged after {it + 1} iterations.')
        break

all_X = np.stack(all_X)
all_U = np.stack(all_U)
all_sigma = np.ones(K) * sigma

if not converged:
    print('Maximum number of iterations reached without convergence.')

# save trajectory to file for visualization
save_arrays('output/trajectory/', {'X': all_X, 'U': all_U, 'sigma': all_sigma})

# plot trajectory
plot(all_X, all_U, all_sigma)
