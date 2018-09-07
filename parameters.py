# Trajectory points
K = 40

# Max solver iterations
iterations = 30

# Weight constants
w_nu = 1e5  # virtual control
# initial trust region radius
tr_radius = 5
# trust region variables
rho_0 = 0.0
rho_1 = 0.25
rho_2 = 0.9
alpha = 2.0
beta = 3.2

solver = ['ECOS', 'MOSEK'][0]
verbose_solver = False
