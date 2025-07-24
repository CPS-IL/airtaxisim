import ipdb
import jax.numpy as jnp
from loguru import logger

from jax_guam.utils.jax_types import Af1, Af4, Af6, Af8, AfBool, AfVec


def w_induced(x: Af6, u: Af4, w: Af8, ders: bool, uL: Af1, uL_x, uL_u) -> tuple[Af1, Af6, Af4]:
    """
    :param x: [ u v w p q r]
    :param u: [ T To del_f i ].
    :param w: [ bx by bz A Ao S rho gam ]
    :param ders:
    :param uL:
    :param uL_x:
    :param uL_u:
    :return:

             T - thrust of the non-overlapped propeller, N (lbf)
            To - thrust of the overlapped propeller, N (lbf)
         del_f - deflection angle of the flap, rad
             i - tilt angle of the wing, rad

              bx - strip c/4 x location with respect to center of mass, m (ft)
              by - strip c/4 y location with respect to center of mass, m (ft)
              bz - strip c/4 z location with respect to center of mass, m (ft)
               A - non-overlapped propeller strip area, m^2 (ft^2)
              Ao - overlapped propeller strip area, m^2 (ft^2)
               S - wing strip area, m^2 (ft^2)
             rho - air density, kg/m^3 (slugs/ft^3)
             gam - wing dihedral angle
    """
    # number of strips
    n_strips = x.shape[0]
    assert x.shape == (n_strips, 6) and u.shape == (n_strips, 4)
    assert w.shape == (n_strips, 8) and uL.shape == (n_strips, 1)

    # Thrust
    T: AfVec = u[:, 0]
    To: AfVec = u[:, 1]

    # Propeller disk area
    A: AfVec = w[:, 3]
    Ao: AfVec = w[:, 4]

    # air density
    rho: AfVec = w[:, 6]

    # determine the indices where there is a propeller overlap
    idx = (Ao > 0) & (To > 0)  # indices of overlap propellers
    N_isoverlap: AfBool = idx

    jdx = (A > 0) & (T > 0) & (~N_isoverlap)  # indices of nonoverlapped propellers
    N_notoverlap = jdx

    # logger.info(
    #     "positive idxs: {} / {}, jdx: {} / {}. max: {}, {}, uL={}".format(
    #         (idx >= 0).sum(),
    #         idx.shape,
    #         (jdx >= 0).sum(),
    #         jdx.shape,
    #         idx.max(),
    #         jdx.max(),
    #         uL.shape,
    #     ),
    # )
    # assert idx.max() < len(uL)
    # assert jdx.max() < len(uL)

    # Propeller induced velocity, use the overlapped conditions
    # when there is an overlapped set of propellers
    uL_squeeze = uL.squeeze(-1)
    denom_i = Ao * rho
    denom_j = A * rho
    denom_i = jnp.where(denom_i == 0, 1.0, denom_i)
    denom_j = jnp.where(denom_j == 0, 1.0, denom_j)

    i_sqrt: AfVec = jnp.sqrt(uL_squeeze**2 + 2 * To / denom_i)
    j_sqrt: AfVec = jnp.sqrt(uL_squeeze**2 + 2 * T / denom_j)

    N_is_ij = N_isoverlap | N_notoverlap
    ij_sqrt = jnp.where(N_isoverlap, i_sqrt, j_sqrt)

    wi_ij = 0.5 * (-uL_squeeze + ij_sqrt)

    # wi_overlap = 0.5 * (-uL + i_sqrt)
    # wi_notoverlap = 0.5 * (-uL + j_sqrt)

    # Note: Somehow, its all false for both overlap and not overlap...
    # wi = jnp.where(N_isoverlap, wi_overlap, jnp.where(N_notoverlap, wi_notoverlap, 0))
    wi = jnp.where(N_is_ij, wi_ij, 0)[:, None]
    assert wi.shape == (n_strips, 1)

    # iSqrt = jnp.sqrt(uL[N_isoverlap] ** 2 + 2 * To[N_isoverlap] / (Ao[N_isoverlap] * rho[N_isoverlap]))
    # jSqrt = jnp.sqrt(uL[N_notoverlap] ** 2 + 2 * T[N_notoverlap] / (A[N_notoverlap] * rho[N_notoverlap]))

    # wi[idx] = 0.5 * (-uL[idx] + iSqrt)
    # wi[jdx] = 0.5 * (-uL[jdx] + jSqrt)

    # wi = jnp.zeros((n_strips, 1))
    # wi = wi.at[idx].set(0.5 * (-uL[N_isoverlap] + iSqrt))
    # wi = wi.at[jdx].set(0.5 * (-uL[N_notoverlap] + jSqrt))

    # outputs
    y = wi

    # if ders:
    #     # Derivatives with respect to states and inputs are not computed in this code snippet.
    #     y_x = None
    #     y_u = None
    # else:
    y_x = jnp.zeros((n_strips, 6))
    y_u = jnp.zeros((n_strips, 4))

    return y, y_x, y_u
