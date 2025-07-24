import jax.numpy as jnp
from loguru import logger

from jax_guam.classes.base_functions.u_local import u_local
from jax_guam.classes.base_functions.v_local import v_local
from jax_guam.classes.base_functions.w_local import w_local


def prop_thrust(x, u, w, prop_coef, ders):
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
        # logger.info("om: {}, J: {}".format(om, J))
        if len(prop_coef) != 1:
            prop_coef = jnp.expand_dims(prop_coef, axis=0)
        cT = prop_coef[:Np, 0] * J**2 + prop_coef[:Np, 1] * J + prop_coef[:Np, 2]
    else:
        Np = 1
        om = jnp.zeros(1)
        # zidx = om < 0.1
        # cT[zidx] = 0
        cT = 0

    T = cT * rho * (om / (2 * jnp.pi)) ** 2 * D**4
    y = T

    # if ders:
    #     vp_uL = ex
    #     vp_vL = ey
    #     vp_wL = ez
    #     J_vp = 2 * jnp.pi / (D * u)
    #     J_om = -2 * jnp.pi * vp / (D * u ** 2)
    #     cT_J = 2 * prop_coef[:, 0] * J + prop_coef[:, 1]
    #     cT_J = jnp.where(zidx, 0.0, cT_J)
    #     T_cT = rho * (u / (2 * jnp.pi)) ** 2 * D ** 4
    #     T_om = T_cT * cT_J * J_om + 2 / (2 * jnp.pi) * rho * (u / (2 * jnp.pi)) * D ** 4 * cT

    #     T_x1 = T_cT * cT_J * J_vp * (vp_uL * uL_x[:, 0] + vp_vL * vL_x[:, 0] + vp_wL * wL_x[:, 0])
    #     T_x2 = T_cT * cT_J * J_vp * (vp_uL * uL_x[:, 1] + vp_vL * vL_x[:, 1] + vp_wL * wL_x[:, 1])
    #     T_x3 = T_cT * cT_J * J_vp * (vp_uL * uL_x[:, 2] + vp_vL * vL_x[:, 2] + vp_wL * wL_x[:, 2])
    #     T_x4 = T_cT * cT_J * J_vp * (vp_uL * uL_x[:, 3] + vp_vL * vL_x[:, 3] + vp_wL * wL_x[:, 3])
    #     T_x5 = T_cT * cT_J * J_vp * (vp_uL * uL_x[:, 4] + vp_vL * vL_x[:, 4] + vp_wL * wL_x[:, 4])
    #     T_x6 = T_cT * cT_J * J_vp * (vp_uL * uL_x[:, 5] + vp_vL * vL_x[:, 5] + vp_wL * wL_x[:, 5])

    #     T_u1 = T_om

    #     y_x = jnp.column_stack([T_x1, T_x2, T_x3, T_x4, T_x5, T_x6])
    #     y_u = jnp.array([T_u1])
    # else:
    y_x = jnp.zeros((Np, 6))
    y_u = jnp.zeros((Np, 1))

    return y, y_x, y_u
