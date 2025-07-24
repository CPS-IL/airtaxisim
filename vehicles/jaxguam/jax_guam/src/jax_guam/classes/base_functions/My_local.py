import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline


def My_local(x, u, w, ders, cm, cm_x, cm_u, qs, qs_x, qs_u):
    S = jnp.expand_dims(w[:, 5], 1)
    My = cm * S * qs
    y = My

    # if ders:
    # else:
    N = x.shape[0]
    y_x = jnp.zeros((N, 6))
    y_u = jnp.zeros((N, 4))

    return y, y_x, y_u
