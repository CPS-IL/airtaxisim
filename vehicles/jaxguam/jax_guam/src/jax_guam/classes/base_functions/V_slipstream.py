import jax.numpy as jnp


def V_slipstream(x, u, w, ders, uL, uL_x, uL_u, vL, vL_x, vL_u, wL, wL_x, wL_u, wi, wi_x, wi_u):
    Vs = jnp.sqrt((uL + 2 * wi) ** 2 + vL**2 + wL**2)
    y = Vs
    # if ders:
    # else:
    N = x.shape[0]
    y_x = jnp.zeros((N, 6))
    y_u = jnp.zeros((N, 4))

    return y, y_x, y_u
