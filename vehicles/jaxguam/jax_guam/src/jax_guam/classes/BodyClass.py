import jax.numpy as jnp


class BodyClass:
    def __init__(self, m_b, Ib, body_cm_b, S_b, S_p, S_s, f_ld):
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
        self.C_Db = 0.029 / jnp.sqrt(self.C_Dfb)

        self.Fx = 0
        self.Fy = 0
        self.Fz = 0
        self.Mx = 0
        self.My = 0
        self.Mz = 0

        self.Fx_x = jnp.zeros(6)
        self.Fy_x = jnp.zeros(6)
        self.Fz_x = jnp.zeros(6)
        self.Mx_x = jnp.zeros(6)
        self.My_x = jnp.zeros(6)
        self.Mz_x = jnp.zeros(6)

    def aero(self, rho, uvw, om, cm_b, ders):
        # Compute the magnitude of the velocity and its derivatives
        V = jnp.sqrt(jnp.sum(uvw**2))
        if V == 0:
            V_u = 0
            V_v = 0
            V_w = 0
        else:
            V_u = uvw[0] / V
            V_v = uvw[1] / V
            V_w = uvw[2] / V

        # Compute the dynamic pressure and its derivatives
        q = 0.5 * rho * V**2
        q_u = rho * V * V_u
        q_v = rho * V * V_v
        q_w = rho * V * V_w

        # Angle of attack
        alf = jnp.arctan2(uvw[2], uvw[0])

        # Derivatives with respect to the local velocity components
        if V == 0:
            alf_u = 0
            alf_w = 0
        else:
            alf_u = (1 / (1 + (uvw[2] / uvw[0]) ** 2)) * (-uvw[2] / uvw[0] ** 2)
            alf_w = (1 / (1 + (uvw[2] / uvw[0]) ** 2)) * (1 / uvw[0])

        # Z Force Coefficient
        Cz = -jnp.sin(2 * alf) * jnp.cos(alf / 2) - self.eta * self.c_dc * self.S_p / self.S_b * jnp.sin(alf) ** 2
        Cz_alf = (
            -2 * jnp.cos(2 * alf) * jnp.cos(alf / 2)
            + 0.5 * jnp.sin(2 * alf) * jnp.sin(alf / 2)
            - self.eta * self.c_dc * self.S_p / self.S_b * 2 * jnp.cos(alf) * jnp.sin(alf)
        )

        # X Force Coefficient
        Cx = -(self.C_f + self.C_Db) * jnp.cos(alf) ** 2
        Cx_alf = (self.C_f + self.C_Db) * 2 * jnp.sin(alf) * jnp.cos(alf)

        self.Fx = float(q * Cx * self.S_b)
        self.Fy = 0
        self.Fz = float(q * Cz * self.S_b)
        self.Mx = 0
        self.My = 0
        self.Mz = 0

        # if ders:
        #     Fx_q = Cx * self.S_b
        #     Fz_q = Cz * self.S_b

        #     Fx_Cx = q * self.S_b
        #     Fz_Cz = q * self.S_b

        #     Fx_u = Fx_q * q_u + Fx_Cx * Cx_alf * alf_u
        #     Fx_w = Fx_q * q_w + Fx_Cx * Cx_alf * alf_w

        #     Fz_u = Fz_q * q_u + Fz_Cz * Cz_alf * alf_u
        #     Fz_w = Fz_q * q_w + Fz_Cz * Cz_alf * alf_w

        #     self.Fx_x = jnp.array([Fx_u, 0, Fx_w, 0, 0, 0])
        #     self.Fy_x = jnp.zeros(6)
        #     self.Fz_x = jnp.array([Fz_u, 0, Fz_w, 0, 0, 0])

        #     self.Mx_x = jnp.zeros(6)
        #     self.My_x = jnp.zeros(6)
        #     self.Mz_x = jnp.zeros(6)
        return self
