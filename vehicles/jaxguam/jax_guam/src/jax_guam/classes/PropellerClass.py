import ipdb
import jax.numpy as jnp

from jax_guam.classes.base_functions.prop_thrust import prop_thrust
from jax_guam.classes.base_functions.prop_torque import prop_torque
from jax_guam.utils.debug import log_local_shapes


class PropellerClass:
    def __init__(self, coef, spin, Dp, p_b, m_b, motor_mass, e_b):
        self.coef = coef
        self.spin = spin
        self.Dp = Dp
        self.p_b = jnp.expand_dims(p_b, axis=1)
        self.m_b = m_b
        self.cm_b = self.m_b
        self.motor_mass = motor_mass
        self.mass = self.motor_mass
        self.e_b = jnp.expand_dims((e_b / jnp.linalg.norm(e_b)), axis=1)
        self.om_prop = None  # TODO this is not defined anywhere, print its value when rerun

        self.T = 0
        self.Q = 0

        self.Fx = 0
        self.Fy = 0
        self.Fz = 0
        self.Mx = 0
        self.My = 0
        self.Mz = 0

        # All propellers have identity orientation (i.e., pointing up in body frame?).
        self.yaw = 0
        self.pitch = 0
        self.roll = 0

        self.Fx_x = jnp.zeros(6)
        self.Fy_x = jnp.zeros(6)
        self.Fz_x = jnp.zeros(6)
        self.Mx_x = jnp.zeros(6)
        self.My_x = jnp.zeros(6)
        self.Mz_x = jnp.zeros(6)

        self.Fx_u = jnp.zeros(4)
        self.Fy_u = jnp.zeros(4)
        self.Fz_u = jnp.zeros(4)
        self.Mx_u = jnp.zeros(4)
        self.My_u = jnp.zeros(4)
        self.Mz_u = jnp.zeros(4)

    def Rx(self, x):
        return jnp.array([[1, 0, 0], [0, jnp.cos(x), -jnp.sin(x)], [0, jnp.sin(x), jnp.cos(x)]])

    def Ry(self, x):
        return jnp.array([[jnp.cos(x), 0, jnp.sin(x)], [0, 1, 0], [-jnp.sin(x), 0, jnp.cos(x)]])

    def Rz(self, x):
        return jnp.array([[jnp.cos(x), -jnp.sin(x), 0], [jnp.sin(x), jnp.cos(x), 0], [0, 0, 1]])

    def hat(self, x):
        return jnp.array(
            [[0, -float(x[2]), float(x[1])], [float(x[2]), 0, -float(x[0])], [float(-x[1]), float(x[0]), 0]]
        )

    def p_c(self, cm_b):
        return self.p_b - cm_b

    def aero(self, rho, uvw, om, cm_b, ders: bool):
        assert isinstance(ders, bool)
        # Get the propeller location in the velocity frame
        p_c = self.p_c(cm_b)

        # Compute the thrust vector
        e = self.Rx(self.roll) @ self.Ry(self.pitch) @ self.Rz(self.yaw) @ self.e_b

        w = jnp.concatenate([p_c.T, e.T, jnp.array([[self.Dp]]), jnp.array([[rho]])], axis=1)
        x = jnp.concatenate([uvw.T, om.T], axis=1)
        u = self.om_prop

        # Note: Q is also used from self.Prop[i].Q in self.get_Qp() in TiltWingClass.py
        if u == 0:
            T = 0
            Q = 0
        else:
            T, T_x, T_om = prop_thrust(x, u, w, self.coef[:, 0], ders)
            Q, Q_x, Q_om = prop_torque(x, u, w, self.coef[:, 1], ders)

        self.T = float(T)
        self.Q = float(Q)

        F = T * e
        M = -self.hat(F) @ p_c + self.spin * Q * e

        self.Fx = F[0]
        self.Fy = F[1]
        self.Fz = F[2]
        self.Mx = M[0]
        self.My = M[1]
        self.Mz = M[2]
        # import pdb; pdb.set_trace()

        # if ders:
        #     Sx = jnp.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        #     e_roll = Sx @ self.Rx(self.roll) @ self.Ry(self.pitch) @ self.Rz(self.yaw) @ self.e_b

        #     Sy = jnp.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        #     e_pitch = self.Rx(self.roll) @ Sy @ self.Ry(self.pitch) @ self.Rz(self.yaw) @ self.e_b

        #     Sz = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        #     e_yaw = self.Rx(self.roll) @ self.Ry(self.pitch) @ Sz @ self.Rz(self.yaw) @ self.e_b

        #     F_T = e
        #     F_om = F_T @ T_om
        #     F_roll = T * e_roll
        #     F_pitch = T * e_pitch
        #     F_yaw = T * e_yaw

        #     M_T = -self.hat(F_T) @ p_c
        #     M_Q = self.spin * e
        #     M_om = M_T @ T_om + M_Q @ Q_om
        #     M_roll = -self.hat(F_roll) @ p_c
        #     M_pitch = -self.hat(F_pitch) @ p_c
        #     M_yaw = -self.hat(F_yaw) @ p_c

        #     self.Fx_x = F_T[0] * T_x
        #     self.Fy_x = F_T[1] * T_x
        #     self.Fz_x = F_T[2] * T_x
        #     self.Mx_x = M_T[0] * T_x + M_Q[0] * Q_x
        #     self.My_x = M_T[1] * T_x + M_Q[1] * Q_x
        #     self.Mz_x = M_T[2] * T_x + M_Q[2] * Q_x

        #     self.Fx_u = jnp.array([F_om[0], F_roll[0], F_pitch[0], F_yaw[0]])
        #     self.Fy_u = jnp.array([F_om[1], F_roll[1], F_pitch[1], F_yaw[1]])
        #     self.Fz_u = jnp.array([F_om[2], F_roll[2], F_pitch[2], F_yaw[2]])
        #     self.Mx_u = jnp.array([M_om[0], M_roll[0], M_pitch[0], M_yaw[0]])
        #     self.My_u = jnp.array([M_om[1], M_roll[1], M_pitch[1], M_yaw[1]])
        #     self.Mz_u = jnp.array([M_om[2], M_roll[2], M_pitch[2], M_yaw[2]])
        return self
