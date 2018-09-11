import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits import mplot3d
from Models.diffdrive_2d import DiffDrive2D

m = DiffDrive2D()


def my_plot(fig, figures_i):
    ax = fig.add_subplot(111)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    X_i = X[figures_i, :, :]
    U_i = U[figures_i, :, :]
    K = X_i.shape[1]

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    ax.set_title("iter " + str(figures_i))

    ax.plot(X_i[0, :], X_i[1, :], color='lightgrey', zorder=0)

    for k in range(K):
        heading = X_i[2, k]

        x = X_i[0, k]
        y = X_i[1, k]

        ax.arrow(x, y, np.cos(heading) * m.robot_radius, np.sin(heading) * m.robot_radius, color='green',
                 head_width=0.2, length_includes_head=True)

        robot = plt.Circle((x, y), m.robot_radius, color='gray', fill=False)
        ax.add_artist(robot)

    for obst in m.obstacles:
        x, y = obst[0]
        r = obst[1]

        obstacle1 = plt.Circle((x, y), r, color='black', fill=False)
        ax.add_artist(obstacle1)


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


def plot2d(X_in, U_in):
    global figures_i, figures_N
    figures_N = X_in.shape[0]
    figures_i = figures_N - 1

    global X, U
    X = X_in
    U = U_in

    fig = plt.figure(figsize=(10, 12))
    my_plot(fig, figures_i)
    cid = fig.canvas.mpl_connect('key_press_event', key_press_event)
    plt.show()


if __name__ == "__main__":
    import os

    folder_number = str(int(max(os.listdir('output/trajectory/')))).zfill(3)

    X_in = np.load(f"output/trajectory/{folder_number}/X.npy")
    U_in = np.load(f"output/trajectory/{folder_number}/U.npy")

    plot2d(X_in, U_in)
