import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import collections as mc
from mpl_toolkits import mplot3d
from Models.diffdrive_2d import Model

m = Model()


def my_plot(fig, figures_i):
    X_i = X[figures_i, :, :]
    U_i = U[figures_i, :, :]
    K = X_i.shape[1]

    gs1 = gridspec.GridSpec(nrows=1, ncols=1, left=0.05, right=0.95, top=0.95, bottom=0.35)
    gs2 = gridspec.GridSpec(nrows=1, ncols=2, left=0.1, right=0.9, top=0.3, bottom=0.05)

    ax = fig.add_subplot(gs1[0, 0])
    ax.set_xlim(m.lower_bound, m.upper_bound)
    ax.set_ylim(m.lower_bound, m.upper_bound)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')

    ax.set_title("Iteration " + str(figures_i))

    ax.plot(X_i[0, :], X_i[1, :], color='lightgrey', zorder=0)

    for k in range(K):
        heading = X_i[2, k]

        x = X_i[0, k]
        y = X_i[1, k]
        dx = np.cos(heading) * m.robot_radius
        dy = np.sin(heading) * m.robot_radius

        ax.arrow(x, y, dx, dy, color='green', head_width=0.2, length_includes_head=True)

        robot = plt.Circle((x, y), m.robot_radius, color='gray', fill=False)
        ax.add_artist(robot)

    for obst in m.obstacles:
        x, y = obst[0]
        r = obst[1]

        obstacle1 = plt.Circle((x, y), r, color='black', fill=False)
        ax.add_artist(obstacle1)

    ax = fig.add_subplot(gs2[0, 0])
    x = np.linspace(0, sigma[figures_i], K)
    ax.plot(x, U_i[0, :])
    ax.set_xlabel('time [s]')
    ax.set_ylabel('velocity [m/s]')

    ax = fig.add_subplot(gs2[0, 1])
    ax.plot(x, np.rad2deg(U_i[1, :]))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('angular velocity [°/s]')


def key_press_event(event):
    global figures_i, figures_N

    fig = event.canvas.figure

    if event.key == 'q' or event.key == 'escape':
        plt.close(event.canvas.figure)
        return
    if event.key == 'right':
        figures_i += 1
        figures_i %= figures_N
    elif event.key == 'left':
        figures_i -= 1
        figures_i %= figures_N
    fig.clear()
    my_plot(fig, figures_i)
    plt.draw()


def plot(X_in, U_in, sigma_in):
    global figures_i, figures_N
    figures_N = X_in.shape[0]
    figures_i = figures_N - 1

    global X, U, sigma
    X = X_in
    U = U_in
    sigma = sigma_in

    fig = plt.figure(figsize=(10, 12))
    my_plot(fig, figures_i)
    cid = fig.canvas.mpl_connect('key_press_event', key_press_event)
    plt.show()


if __name__ == "__main__":
    import os

    folder_number = str(int(max(os.listdir('output/trajectory/')))).zfill(3)

    X_in = np.load(f"output/trajectory/{folder_number}/X.npy")
    U_in = np.load(f"output/trajectory/{folder_number}/U.npy")
    sigma_in = np.load(f"output/trajectory/{folder_number}/sigma.npy")

    plot(X_in, U_in, sigma_in)
