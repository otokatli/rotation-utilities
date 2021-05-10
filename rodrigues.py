from math import cos, sin
from numpy import array, eye
from numpy.linalg import matrix_power, norm


def skew(k):
    return array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])


def normalize(k):
    return k / norm(k)


def rodrigues(k, theta):
    """
    Rodrigues rotation formula

    :param k: Axis of rotation
    :param theta: Angle of ratation
    :return: Rotation matrix associated with (v, theta)
    """

    k_normalized = normalize(k)

    return eye(3) + sin(theta) * skew(k_normalized) + (1 - cos(theta)) \
           * matrix_power(skew(k_normalized), 2)
