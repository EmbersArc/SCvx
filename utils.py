import numpy as np
import os


def euler_to_quat(a):
    a = np.deg2rad(a)

    cy = np.cos(a[1] * 0.5)
    sy = np.sin(a[1] * 0.5)
    cr = np.cos(a[0] * 0.5)
    sr = np.sin(a[0] * 0.5)
    cp = np.cos(a[2] * 0.5)
    sp = np.sin(a[2] * 0.5)

    q = np.zeros(4)

    q[0] = cy * cr * cp + sy * sr * sp
    q[1] = cy * sr * cp - sy * cr * sp
    q[3] = cy * cr * sp + sy * sr * cp
    q[2] = sy * cr * cp - cy * sr * sp

    return q


def format_line(name, value, unit=''):
    """
    Formats a line e.g.
    {Name:}           {value}{unit}
    """
    name += ':'
    if isinstance(value, (float, np.ndarray)):
        value = f'{value:{0}.{4}}'

    return f'{name.ljust(40)}{value}{unit}'


def save_arrays(path, a_dict):
    """
    :param path: Output path
    :param a_dict: A dict containing the name of the array as key.
    """
    path = path.rstrip('/')

    if not os.path.isdir(path):
        os.mkdir(path)

    if len(os.listdir(path)) == 0:
        folder_number = '000'
    else:
        folder_number = str(int(max(os.listdir(path))) + 1).zfill(3)

    os.mkdir(f'{path}/{folder_number}')

    for key in a_dict:
        np.save(f'{path}/{folder_number}/{key}.npy', a_dict[key])
