import numpy as np


def linear_kernel(x, offset: float, width: float):
    """Linear interpolation kernel"""
    # x = x - offset
    # x = x / width
    return (1 - np.abs((x - offset) / width)) * (np.abs((x - offset) / width) < 1)


def rectangular_kernel(t):
    return np.where((t >= 0) & (t < 1), 1, 0)


def h2_kernel(t):
    return np.where((t >= -0.5) & (t < 0.5), 1, 0)


def h3_kernel(t):
    return np.where((t >= -1) & (t < 1), 1 - abs(t), 0)


def sin_kernel(x):
    return np.where(x == 0, 1, np.sin(x)/x)


def sinc_kernel(x):
    return np.sinc(x/np.pi)


def cubic_kernel(t):
    condition1 = (0 < np.abs(t)) & (np.abs(t) < 1)
    condition2 = (1 < np.abs(t)) & (np.abs(t) < 2)

    result = np.where(condition1, (3 * pow(np.abs(t), 3)) / 2 - (5 * pow(np.abs(t), 2)) / 2 + 1,
                      np.where(condition2, (-pow(np.abs(t), 3)) / 2 + (5 * pow(np.abs(t), 2)) / 2 - (4 * np.abs(t)) + 2, 0))

    return result
