import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits import mplot3d

figures_i = 0

# vector scaling
thrust_scale = 0.00002
attitude_scale = 20


def key_press_event(event):
    global figures_i
    fig = event.canvas.figure

    if event.key == 'q' or event.key == 'escape':
        plt.close(event.canvas.figure)
        return

    if event.key == 'right':
        figures_i = (figures_i + 1) % figures_N
    elif event.key == 'left':
        figures_i = (figures_i - 1) % figures_N

    fig.clear()
    my_plot(fig, figures_i)
    plt.draw()


def my_plot(fig, figures_i):
    ax = fig.add_subplot(111, projection='3d')

    X_i = X[figures_i, :, :]
    U_i = U[figures_i, :, :]
    K = X_i.shape[1]

    ax.set_zlabel('X, up')
    ax.set_xlabel('Y, east')
    ax.set_ylabel('Z, north')

    for k in range(K):
        rx, ry, rz = X_i[1:4, k]
        vx, vy, vz = X_i[4:7, k]
        qw, qx, qy, qz = X_i[7:11, k]

        CBI = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
            [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        Fx, Fy, Fz = np.dot(np.transpose(CBI), U_i[:, k])
        dx, dy, dz = np.dot(np.transpose(CBI), np.array([1., 0., 0.]))

        # # speed vector
        # ax.quiver(ry, rz, rx, vy, vz, vx, length=0.1, color='green')

        # attitude vector
        ax.quiver(ry, rz, rx, dy, dz, dx, length=attitude_scale, arrow_length_ratio=0.0, color='blue')

        # thrust vector
        ax.quiver(ry, rz, rx, -Fy, -Fz, -Fx, length=thrust_scale, arrow_length_ratio=0.0, color='red')

    scale = X_i[1, 0]
    ax.auto_scale_xyz([-scale/2, scale/2], [-scale/2, scale/2], [0, scale])

    ax.set_title("Iteration " + str(figures_i))
    ax.plot(X_i[2, :], X_i[3, :], X_i[1, :], color='black')


def plot(X_in, U_in, sigma_in):
    global figures_N
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
    sigma_in = np.load(f"output/trajectory/{folder_number}/sigma.npy")

    plot(X_in, U_in, sigma_in)
