import jax.numpy as jnp


def alf_slipstream(x, u, w, ders, uL, uL_x, uL_u, wL, wL_x, wL_u, wi, wi_x, wi_u):
    alfs = jnp.arctan2(wL, uL + 2 * wi)
    y = alfs
    # if ders:
    # else:
    N = x.shape[0]
    y_x = jnp.zeros((N, 6))
    y_u = jnp.zeros((N, 4))

    return y, y_x, y_u
