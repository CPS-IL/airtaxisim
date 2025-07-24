import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline


def Fz_local(x, u, w, ders, cl, cl_x, cl_u, cd, cd_x, cd_u, alfs, alfs_x, alfs_u, qs, qs_x, qs_u):
    T = jnp.expand_dims(u[:, 0], 1)
    To = jnp.expand_dims(u[:, 1], 1)
    A = jnp.expand_dims(w[:, 3], 1)
    Ao = jnp.expand_dims(w[:, 4], 1)
    iw = jnp.expand_dims(u[:, 3], 1)
    S = jnp.expand_dims(w[:, 5], 1)
    c_iw = jnp.cos(iw)
    s_iw = jnp.sin(iw)
    c_iw_alfs = jnp.cos(iw - alfs)
    s_iw_alfs = jnp.sin(iw - alfs)
    Fz = -T * s_iw + To * s_iw - (cl * c_iw_alfs - cd * s_iw_alfs) * S * qs
    y = Fz

    # if ders:
    # else:
    N = x.shape[0]
    y_x = jnp.zeros((N, 6))
    y_u = jnp.zeros((N, 4))

    return y, y_x, y_u
