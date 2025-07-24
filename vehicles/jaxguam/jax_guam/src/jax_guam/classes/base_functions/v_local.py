import jax.numpy as jnp

from jax_guam.utils.jax_types import RowVec6, RowVec4, RowVec8


def v_local(x: RowVec6, u: RowVec4, w: RowVec8, ders: bool):
    assert isinstance(ders, bool) and ders is False
    n_points = len(x)
    assert x.shape == (n_points, 6) and u.shape == (n_points, 4) and w.shape == (n_points, 8)
    N = x.shape[0]

    uu = x[:, 0]
    vv = x[:, 1]
    ww = x[:, 2]
    p = x[:, 3]
    q = x[:, 4]
    r = x[:, 5]

    T = u[:, 0]
    To = u[:, 1]
    del_f = u[:, 2]
    i = u[:, 3]

    bx = w[:, 0]
    by = w[:, 1]
    bz = w[:, 2]
    gam = w[:, 7]

    c_gam = jnp.cos(gam)
    s_gam = jnp.sin(gam)

    y = c_gam * (vv - p * bz + r * bx) + s_gam * (ww + p * by - q * bx)

    if ders:
        y_x = jnp.column_stack(
            [jnp.zeros(N), c_gam, s_gam, -c_gam * bz + s_gam * by - s_gam * bx, -s_gam * bx, c_gam * bx]
        )

        y_u = jnp.zeros((N, 4))
    else:
        y_x = jnp.zeros((N, 6))
        y_u = jnp.zeros((N, 4))

    return jnp.expand_dims(y, 1), y_x, y_u
