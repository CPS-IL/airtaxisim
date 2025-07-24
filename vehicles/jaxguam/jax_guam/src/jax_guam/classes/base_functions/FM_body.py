from typing import NamedTuple

import jax.numpy as jnp

from jax_guam.classes.base_functions.alf_flaps import alf_flaps
from jax_guam.classes.base_functions.alf_slipstream import alf_slipstream
from jax_guam.classes.base_functions.cd_local import cd_local
from jax_guam.classes.base_functions.cl_local import cl_local
from jax_guam.classes.base_functions.cm_local import cm_local
from jax_guam.classes.base_functions.Fx_local import Fx_local
from jax_guam.classes.base_functions.Fz_local import Fz_local
from jax_guam.classes.base_functions.My_local import My_local
from jax_guam.classes.base_functions.q_slipstream import q_slipstream
from jax_guam.classes.base_functions.u_local import u_local
from jax_guam.classes.base_functions.v_local import v_local
from jax_guam.classes.base_functions.V_slipstream import V_slipstream
from jax_guam.classes.base_functions.w_induced import w_induced
from jax_guam.classes.base_functions.w_local import w_local
from jax_guam.utils.debug import log_local_shapes
from jax_guam.utils.jax_types import Af1, Af4, Af6, Af8


class FMBodyOut(NamedTuple):
    FM: Af6
    FM_x: Af6
    FM_u: Af4


def FM_body(x: Af6, u: Af4, w: Af8, aero_coefs_pp, ders: bool) -> FMBodyOut:
    assert isinstance(ders, bool) and ders is False
    n_points = x.shape[0]

    gam = jnp.expand_dims(w[:, 7], 1)

    bx = jnp.expand_dims(w[:, 0], 1)
    by = jnp.expand_dims(w[:, 1], 1)
    bz = jnp.expand_dims(w[:, 2], 1)

    uL, uL_x, uL_u = u_local(x, u, w, ders)
    vL, vL_x, vL_u = v_local(x, u, w, ders)
    wL, wL_x, wL_u = w_local(x, u, w, ders)

    wi, wi_x, wi_u = w_induced(x, u, w, ders, uL, uL_x, uL_u)

    alfs, alfs_x, alfs_u = alf_slipstream(x, u, w, ders, uL, uL_x, uL_u, wL, wL_x, wL_u, wi, wi_x, wi_u)

    alff, alff_x, alff_u = alf_flaps(x, u, w, ders, alfs, alfs_x, alfs_u)

    cl, cl_x, cl_u = cl_local(x, u, w, aero_coefs_pp, ders, alff, alff_x, alff_u)

    cd, cd_x, cd_u = cd_local(x, u, w, aero_coefs_pp, ders, alff, alff_x, alff_u)

    cm, cm_x, cm_u = cm_local(x, u, w, aero_coefs_pp, ders, alff, alff_x, alff_u)

    Vs, Vs_x, Vs_u = V_slipstream(x, u, w, ders, uL, uL_x, uL_u, vL, vL_x, vL_u, wL, wL_x, wL_u, wi, wi_x, wi_u)

    qs, qs_x, qs_u = q_slipstream(x, u, w, ders, Vs, Vs_x, Vs_u)

    Fxl, Fxl_x, Fxl_u = Fx_local(x, u, w, ders, cl, cl_x, cl_u, cd, cd_x, cd_u, alfs, alfs_x, alfs_u, qs, qs_x, qs_u)

    Fzl, Fzl_x, Fzl_u = Fz_local(x, u, w, ders, cl, cl_x, cl_u, cd, cd_x, cd_u, alfs, alfs_x, alfs_u, qs, qs_x, qs_u)

    Myl, Myl_x, Myl_u = My_local(x, u, w, ders, cm, cm_x, cm_u, qs, qs_x, qs_u)
    Fx: Af1 = Fxl
    Fy: Af1 = -Fzl * jnp.sin(gam)
    Fz: Af1 = Fzl * jnp.cos(gam)
    Mx: Af1 = Fz * by - Fy * bz
    My: Af1 = Myl * jnp.cos(gam) - Fz * bx + Fx * bz
    Mz: Af1 = Myl * jnp.sin(gam) - Fx * by + Fy * bx
    # FM: Af6 = jnp.column_stack([Fx, Fy, Fz, Mx, My, Mz])
    FM: Af6 = jnp.concatenate([Fx, Fy, Fz, Mx, My, Mz], axis=1)
    assert FM.shape == (n_points, 6)
    # if ders:

    # else:
    FM_x = jnp.zeros((n_points, 6))
    FM_u = jnp.zeros((n_points, 4))

    return FMBodyOut(FM, FM_x, FM_u)
