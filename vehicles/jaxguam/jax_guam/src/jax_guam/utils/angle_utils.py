import jax.numpy as jnp
import numpy as np

from jax_guam.utils.jax_types import FloatScalar, Mat3, Vec3


def Rx(phi: FloatScalar) -> Mat3:
    if isinstance(phi, jnp.ndarray):
        c, s = jnp.cos(phi), jnp.sin(phi)
        return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(theta: FloatScalar) -> Mat3:
    if isinstance(theta, jnp.ndarray):
        c, s = jnp.cos(theta), jnp.sin(theta)
        return jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(psi: FloatScalar) -> Mat3:
    if isinstance(psi, jnp.ndarray):
        c, s = jnp.cos(psi), jnp.sin(psi)
        return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def so3_hat(w: Vec3) -> Mat3:
    assert w.shape == (3,)
    if isinstance(w, jnp.ndarray):
        omega_mat = jnp.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    else:
        omega_mat = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    assert omega_mat.shape == (3, 3)
    return omega_mat
