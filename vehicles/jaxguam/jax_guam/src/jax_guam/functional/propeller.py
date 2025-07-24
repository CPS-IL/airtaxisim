from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from jax_guam.classes.base_functions.prop_thrust import prop_thrust
from jax_guam.classes.base_functions.prop_torque import prop_torque
from jax_guam.utils.angle_utils import Rx, Ry, Rz, so3_hat
from jax_guam.utils.jax_types import FloatScalar, FMVec, Mat1, Mat3_2, Vec3, Vec3_1


class FuncPropeller(NamedTuple):
    coef: Mat3_2
    spin: float
    Dp: float
    p_b: Vec3_1
    m_b: Vec3
    cm_b: Vec3
    motor_mass: float
    mass: float
    e_b: Vec3_1

    @staticmethod
    def create(coef: Mat3_2, spin: float, Dp: float, p_b: Vec3, m_b: Vec3, motor_mass: float, e_b: Vec3_1):
        # coef = coef
        # spin = spin
        # Dp = Dp
        p_b = np.expand_dims(p_b, axis=1)
        # m_b = m_b
        cm_b = m_b
        # motor_mass = motor_mass
        mass = motor_mass
        e_b = np.expand_dims((e_b / np.linalg.norm(e_b)), axis=1)

        return FuncPropeller(coef, spin, Dp, p_b, m_b, cm_b, motor_mass, mass, e_b)

        # All propellers have identity orientation (i.e., pointing up in body frame?).
        # self.yaw = 0
        # self.pitch = 0
        # self.roll = 0

    @property
    def yaw(self):
        return 0

    @property
    def pitch(self):
        return 0

    @property
    def roll(self):
        return 0

    def p_c(self, cm_b):
        return self.p_b - cm_b

    def aero(
        self, rho: float, uvw: Vec3_1, om: Vec3_1, cm_b: Vec3_1, ders: bool, om_prop: FloatScalar
    ) -> tuple[FMVec, Mat1, Mat1]:
        assert isinstance(ders, bool) and ders is False
        # Get the propeller location in the velocity frame
        p_c: Vec3_1 = self.p_c(cm_b)

        # Compute the thrust vector
        e = Rx(self.roll) @ Ry(self.pitch) @ Rz(self.yaw) @ self.e_b

        w = jnp.concatenate([p_c.T, e.T, jnp.array([[self.Dp]]), jnp.array([[rho]])], axis=1)
        x = jnp.concatenate([uvw.T, om.T], axis=1)
        u = om_prop

        # if u == 0:
        #     T = 0
        #     Q = 0
        # else:
        # Note: u=0 for the pusher. In this case, we set T = Q = 0. Set u to a dummy value just to avoid the nan.
        u_is_zero = u == 0
        safe_u = jnp.where(u_is_zero, 1.0, u)
        T, T_x, T_om = prop_thrust(x, safe_u, w, self.coef[:, 0], ders)
        Q, Q_x, Q_om = prop_torque(x, safe_u, w, self.coef[:, 1], ders)

        T = jnp.where(u_is_zero, jnp.zeros((1, 1)), T)
        Q = jnp.where(u_is_zero, jnp.zeros((1, 1)), Q)

        assert T.shape == Q.shape == (1, 1)

        F = (T * e).squeeze(-1)
        M = -so3_hat(F) @ p_c + self.spin * Q * e
        M = M.squeeze(-1)
        assert F.shape == M.shape == (3,)

        FM = jnp.concatenate([F, M], axis=0)
        assert FM.shape == (6,)

        return FM, T, Q

        # self.Fx = F[0]
        # self.Fy = F[1]
        # self.Fz = F[2]
        # self.Mx = M[0]
        # self.My = M[1]
        # self.Mz = M[2]
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
