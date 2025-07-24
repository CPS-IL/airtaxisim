import jax.numpy as jnp


def q_slipstream(x, u, w, ders, Vs, Vs_x, Vs_u):
    rho = jnp.expand_dims(w[:, 6], 1)
    qs = 0.5 * rho * Vs**2
    y = qs
    # if ders:
    # else:
    N = x.shape[0]
    y_x = jnp.zeros((N, 6))
    y_u = jnp.zeros((N, 4))

    return y, y_x, y_u
