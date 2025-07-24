import ipdb
import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline

from jax_guam.utils.jax_types import Af1, Af6


def cl_local(x: Af6, u, w, aero_coefs_pp, ders, alff: Af1, alff_x, alff_u):
    # tck = UnivariateSpline._eval_args(aero_coefs_pp.cl)
    # tck = (aero_coefs_pp[0][0][1][0][0][1].T[:-1], aero_coefs_pp[0][0][1][0][0][2], 3)
    # pp_mk = UnivariateSpline._from_tck(tck)
    # cl = pp_mk(alff)
    # import pdb; pdb.set_trace()

    n_strips = len(x)
    assert x.shape == (n_strips, 6) and alff.shape == (n_strips, 1)

    breaks = aero_coefs_pp[0][0][1][0][0][1]
    assert breaks.shape == (1, 541)

    coeffs = aero_coefs_pp[0][0][1][0][0][2]
    assert coeffs.shape == (540, 4)

    inds = jnp.digitize(alff, breaks[0], right=True) - 1
    inds = jnp.clip(inds, 0, len(coeffs) - 1)
    assert inds.shape == (n_strips, 1)

    coeffs = jnp.array(coeffs)[inds]
    assert coeffs.shape == (n_strips, 1, 4)

    alff_min_breaks = alff - jnp.array(breaks)[:, inds]
    assert alff_min_breaks.shape == (1, n_strips, 1)
    # (n_strips, 1) * (1, n_strips, 1) = (1, n_strips, 1)
    y = (
        coeffs[:, :, 0] * alff_min_breaks**3
        + coeffs[:, :, 1] * alff_min_breaks**2
        + coeffs[:, :, 2] * alff_min_breaks**1
        + coeffs[:, :, 3]
    )
    assert y.shape == (1, n_strips, 1)
    # if ders:
    # else:
    N = x.shape[0]
    y_x = jnp.zeros((N, 6))
    y_u = jnp.zeros((N, 4))

    y = y[0]
    assert y.shape == (n_strips, 1)

    return y, y_x, y_u
