import numpy as np
from scipy.special import spherical_yn, spherical_jn
import matplotlib.pyplot as plt


# gradient force = 0
def grad_rho(rho, theta, params):
    l, n_w, x_r = params["l"], params["n_w"], params["x_r"]
    kR = n_w * x_r
    kr = 2 * np.pi * n_w * rho
    value = (
        np.sin(theta) ** (2 * (l - 1))
        * (1 + np.cos(theta) ** 2)
        * (
            spherical_jn(l, kr) * spherical_jn(l, kr, True)
            + spherical_yn(l, kr) * spherical_yn(l, kr, True)
        )
        / (
            spherical_jn(l, kR) * spherical_jn(l, kR, True)
            + spherical_yn(l, kR) * spherical_yn(l, kR, True)
        )
    )
    return 0


def grad_theta(rho, theta, params):
    l, n_w, x_r = params["l"], params["n_w"], params["x_r"]
    kR = n_w * x_r
    kr = 2 * np.pi * n_w * rho
    value = (
        np.sin(theta) ** (2 * l - 3)
        * ((l - 1) * (1 + np.cos(theta) ** 2) - np.sin(theta) ** 2 * np.cos(theta))
        * (spherical_jn(l, kr) ** 2 + spherical_yn(l, kr) ** 2)
        / (
            kr
            * (
                spherical_jn(l, kR) * spherical_jn(l, kR, True)
                + spherical_yn(l, kR) * spherical_yn(l, kR, True)
            )
        )
    )
    return 0


# scattering force = 0
def scat_rho(rho, theta, params):
    l, n_w, n_p, R_ratio, x_r = (
        params["l"],
        params["n_w"],
        params["n_p"],
        params["R_ratio"],
        params["x_r"],
    )
    kR = n_w * x_r
    x = 2 * np.pi * rho
    coeff = (
        (2 / 3)
        * n_w
        * (x_r**3 / x**2)
        * R_ratio**3
        * ((n_p**2 - n_w**2) / (n_p**2 + 2 * n_w**2))
        * np.sin(theta) ** (2 * l - 2)
        * (1 + np.cos(theta) ** 2)
    )
    value = coeff / (
        spherical_jn(l, kR) * spherical_jn(l, kR, True)
        + spherical_yn(l, kR) * spherical_yn(l, kR, True)
    )
    return 0


def scat_phi(rho, theta, params):
    l, n_w, n_p, R_ratio, x_r = (
        params["l"],
        params["n_w"],
        params["n_p"],
        params["R_ratio"],
        params["x_r"],
    )
    kR = n_w * x_r
    x = 2 * np.pi * rho
    kr = n_w * x
    coeff = (
        (2 / 3)
        * n_w**2
        * (x_r**3 / x)
        * R_ratio**3
        * ((n_p**2 - n_w**2) / (n_p**2 + 2 * n_w**2))
        * np.sin(theta) ** (2 * l - 3)
        * (l * (1 + np.sin(theta)) - np.cos(2 * theta))
    )
    value = (
        coeff
        * (spherical_jn(l, kr) ** 2 + spherical_yn(l, kr) ** 2)
        / (
            spherical_jn(l, kR) * spherical_jn(l, kR, True)
            + spherical_yn(l, kR) * spherical_yn(l, kR, True)
        )
    )
    return 0


# combined forces
def rho(rho, theta, params):
    return 1  # grad_rho(rho, theta, params) + scat_rho(rho, theta, params)


def phi(rho, theta, params):
    return 0  # scat_phi(rho, theta, params)


def theta(rho, theta, params):
    return 0  # grad_theta(rho, theta, params)
