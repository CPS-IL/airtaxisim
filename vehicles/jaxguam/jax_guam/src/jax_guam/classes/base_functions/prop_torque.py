import ipdb
import jax.numpy as jnp
import numpy as np

from jax_guam.classes.base_functions.u_local import u_local
from jax_guam.classes.base_functions.v_local import v_local
from jax_guam.classes.base_functions.w_local import w_local
from jax_guam.utils.jax_types import FloatScalar, RowVec6, RowVec8, Vec3, Mat1


def prop_torque(x: RowVec6, u: FloatScalar, w: RowVec8, prop_coef: Vec3, ders: bool) -> tuple[Mat1, RowVec6, Mat1]:
    assert isinstance(ders, bool) and ders is False
    assert x.shape == (1, 6) and w.shape == (1, 8) and prop_coef.shape == (3,)

    ex = w[:, 3]
    ey = w[:, 4]
    ez = w[:, 5]
    D = w[:, 6]
    rho = w[:, 7]

    Np = 1  # len(u)
    if Np > 0:
        om = u  # [:Np, 1]
        xt = x  # [:Np, :6]
        wt = jnp.zeros((Np, 8))
        ut = jnp.zeros((Np, 4))
        # wt[:Np, :3] = w[:Np, :3]
        wt = wt.at[:Np, :3].set(w[:Np, :3])

        uL, uL_x, uL_u = u_local(xt, ut, wt, ders)
        vL, vL_x, vL_u = v_local(xt, ut, wt, ders)
        wL, wL_x, wL_u = w_local(xt, ut, wt, ders)

        vp = ex * uL + ey * vL + ez * wL
        J = 2 * jnp.pi * vp / (D * om)
        if len(prop_coef) != 1:
            prop_coef = jnp.expand_dims(prop_coef, axis=0)
        cP = prop_coef[:Np, 0] * J**2 + prop_coef[:Np, 1] * J + prop_coef[:Np, 2]
        cQ = -cP / (2 * jnp.pi)
    else:
        Np = 1
        om = jnp.ones(1)
        # zidx = om < 0.1
        # cQ[zidx] = 0
        cQ = 0

    Q = cQ * rho * (om / (2 * jnp.pi)) ** 2 * D**5
    y = Q

    # if ders:
    #     vp_uL = ex
    #     vp_vL = ey
    #     vp_wL = ez
    #     J_vp = 2 * jnp.pi / (D * u)
    #     J_om = -2 * jnp.pi * vp / (D * u ** 2)
    #     cP_J = 2 * prop_coef[:, 0] * J + prop_coef[:, 1]
    #     cQ_J = -cP_J / (2 * jnp.pi)

    #     cQ_J = jnp.where(zidx, 0.0, cQ_J)
    #     J_om = jnp.where(zidx, 0.0, J_om)
    #     J_vp = jnp.where(zidx, 0.0, J_vp)

    #     Q_cQ = rho * (u / (2 * jnp.pi)) ** 2 * D ** 5
    #     Q_om = Q_cQ * cQ_J * J_om + 2 / (2 * jnp.pi) * rho * (u / (2 * jnp.pi)) * D ** 5 * cQ

    #     Q_x1 = Q_cQ * cQ_J * J_vp * (vp_uL * uL_x[:, 0] + vp_vL * vL_x[:, 0] + vp_wL * wL_x[:, 0])
    #     Q_x2 = Q_cQ * cQ_J * J_vp * (vp_uL * uL_x[:, 1] + vp_vL * vL_x[:, 1] + vp_wL * wL_x[:, 1])
    #     Q_x3 = Q_cQ * cQ_J * J_vp * (vp_uL * uL_x[:, 2] + vp_vL * vL_x[:, 2] + vp_wL * wL_x[:, 2])
    #     Q_x4 = Q_cQ * cQ_J * J_vp * (vp_uL * uL_x[:, 3] + vp_vL * vL_x[:, 3] + vp_wL * wL_x[:, 3])
    #     Q_x5 = Q_cQ * cQ_J * J_vp * (vp_uL * uL_x[:, 4] + vp_vL * vL_x[:, 4] + vp_wL * wL_x[:, 4])
    #     Q_x6 = Q_cQ * cQ_J * J_vp * (vp_uL * uL_x[:, 5] + vp_vL * vL_x[:, 5] + vp_wL * wL_x[:, 5])

    #     Q_u1 = Q_om

    #     y_x = jnp.column_stack([Q_x1, Q_x2, Q_x3, Q_x4, Q_x5, Q_x6])
    #     y_u = jnp.array([Q_u1])
    # else:
    y_x = jnp.zeros((Np, 6))
    y_u = jnp.zeros((Np, 1))

    return y, y_x, y_u


def prop_torque_jax(x: RowVec6, u: FloatScalar, w: RowVec8, prop_coef: Vec3, ders: bool):
    assert isinstance(ders, bool) and ders is False
    assert x.shape == (1, 6) and w.shape == (1, 8) and prop_coef.shape == (3,)

    ex = w[:, 3]
    ey = w[:, 4]
    ez = w[:, 5]
    D = w[:, 6]
    rho = w[:, 7]

    Np = 1  # len(u)
    assert Np > 0

    om = u  # [:Np, 1]
    xt = x  # [:Np, :6]
    wt = jnp.zeros((Np, 8))
    ut = jnp.zeros((Np, 4))
    # wt[:Np, :3] = w[:Np, :3]
    wt = wt.at[:Np, :3].set(w[:Np, :3])

    uL, uL_x, uL_u = u_local(xt, ut, wt, ders)
    vL, vL_x, vL_u = v_local(xt, ut, wt, ders)
    wL, wL_x, wL_u = w_local(xt, ut, wt, ders)

    vp = ex * uL + ey * vL + ez * wL
    J = 2 * jnp.pi * vp / (D * om)
    if len(prop_coef) != 1:
        prop_coef = jnp.expand_dims(prop_coef, axis=0)

    cP = prop_coef[:Np, 0] * J**2 + prop_coef[:Np, 1] * J + prop_coef[:Np, 2]
    cQ = -cP / (2 * jnp.pi)

    Q = cQ * rho * (om / (2 * jnp.pi)) ** 2 * D**5
    y: Mat1 = Q

    y_x = jnp.zeros((Np, 6))
    y_u = jnp.zeros((Np, 1))

    return y, y_x, y_u
