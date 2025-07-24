import ipdb
import jax.numpy as jnp
import numpy as np

from jax_guam.utils.jax_types import FMVec, Mat3, Vec3_1


class FuncBody:
    def __init__(self, m_b: float, Ib: Mat3, body_cm_b: Vec3_1, S_b: float, S_p: float, S_s: float, f_ld: float):
        self.mass = m_b
        self.I = Ib
        self.cm_b = body_cm_b
        self.S_b = S_b
        self.S_p = S_p
        self.S_s = S_s
        self.f_ld = f_ld

        self.eta = 0.67
        self.c_dc = 1.2
        self.C_f = 0.004

        self.C_Dfb = self.C_f * (1 + 60 / self.f_ld**3 + 0.0025 * self.f_ld) * self.S_s / self.S_b
        self.C_Db = 0.029 / np.sqrt(self.C_Dfb)

    def aero(self, rho: float, uvw: Vec3_1, om: Vec3_1, cm_b: Vec3_1, ders: bool) -> FMVec:
        assert isinstance(ders, bool) and ders is False
        assert uvw.shape == (3, 1)
        # Compute the magnitude of the velocity and its derivatives
        V = jnp.sqrt(1e-14 + jnp.sum(uvw**2))

        # Note: Unused
        # V_uvw = jnp.where(V < 1e-6, jnp.zeros(3), uvw / V)

        # if V == 0:
        #     V_u = 0
        #     V_v = 0
        #     V_w = 0
        # else:
        #     V_u = uvw[0] / V
        #     V_v = uvw[1] / V
        #     V_w = uvw[2] / V

        # Compute the dynamic pressure and its derivatives
        q = 0.5 * rho * V**2

        # Note: Unused
        # q_uvw = rho * V * V_uvw
        # q_u = rho * V * V_u
        # q_v = rho * V * V_v
        # q_w = rho * V * V_w

        # Angle of attack
        alf = jnp.arctan2(uvw[2], uvw[0])

        # Note: Unused
        # # Derivatives with respect to the local velocity components
        # if V == 0:
        #     alf_u = 0
        #     alf_w = 0
        # else:
        #     alf_u = (1 / (1 + (uvw[2] / uvw[0]) ** 2)) * (-uvw[2] / uvw[0] ** 2)
        #     alf_w = (1 / (1 + (uvw[2] / uvw[0]) ** 2)) * (1 / uvw[0])

        # Z Force Coefficient
        Cz = -jnp.sin(2 * alf) * jnp.cos(alf / 2) - self.eta * self.c_dc * self.S_p / self.S_b * jnp.sin(alf) ** 2
        # Cz_alf = (
        #     -2 * jnp.cos(2 * alf) * jnp.cos(alf / 2)
        #     + 0.5 * jnp.sin(2 * alf) * jnp.sin(alf / 2)
        #     - self.eta * self.c_dc * self.S_p / self.S_b * 2 * jnp.cos(alf) * jnp.sin(alf)
        # )

        # X Force Coefficient
        Cx = -(self.C_f + self.C_Db) * jnp.cos(alf) ** 2
        # Cx_alf = (self.C_f + self.C_Db) * 2 * jnp.sin(alf) * jnp.cos(alf)

        FM = jnp.array([q * Cx.squeeze() * self.S_b, 0, q * Cz.squeeze() * self.S_b, 0, 0, 0])
        return FM

        # self.Fx = float(q * Cx * self.S_b)
        # self.Fy = 0
        # self.Fz = float(q * Cz * self.S_b)
        # self.Mx = 0
        # self.My = 0
        # self.Mz = 0
