import jax.numpy as jnp

from jax_guam.utils.debug import log_local_shapes
from jax_guam.utils.jax_types import Af6, Af4, Af8, Af1


def alf_flaps(x: Af6, u: Af4, w: Af8, ders, alfs: Af1, alfs_x, alfs_u) -> tuple[Af1, Af6, Af4]:
    dell = jnp.expand_dims(u[:, 2], 1)
    idx = ~(dell == 255)
    tau = 0.6
    alff = alfs + tau * dell * idx
    y = alff

    # if ders:
    # else:
    N = x.shape[0]
    y_x = jnp.zeros((N, 6))
    y_u = jnp.zeros((N, 4))

    return y, y_x, y_u
