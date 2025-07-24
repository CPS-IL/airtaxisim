import functools as ft
import pdb

import ipdb
import jax
import jax.numpy as jnp
import numpy as np
from extensisq import BS5
from loguru import logger
from scipy.integrate import solve_ivp

from jax_guam.functional.vehicle_eom_utils import get_air_rel, get_Q2hb_AltMSL, get_world_air_rel_simple, get_world_rel
from jax_guam.guam_types import *
from jax_guam.utils.functions import (
    nquaternion_to_Euler,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiplication,
    quaternion_to_DCM,
    quaternion_vector_transformation,
    quaternion_vector_transformation2,
    single_axis_quaternion,
)
from jax_guam.utils.jax_types import FloatScalar, FMVec_1, Force, Mat3, Moment, Vec3_1, Vec6, Vec6_1, Vec13


class VehicleEOMSimple:
    def __init__(self):
        self.MASS = 181.78924904000002
        self.INERTIA = np.array([[13052, 58, -2969], [58, 16661, -986], [-2969, -986, 24735]])
        self.INVINERTIA = np.linalg.inv(self.INERTIA)

    def get_fm_with_gravity(self, state13: AircraftStateVec, fm: FM) -> FMVec_1:
        state_split = AircraftState.from_state13(state13)

        # Add gravity to total fm.
        f_gravity: Vec3_1 = self._get_gravity_force(state_split.QDt_i2b)[:, None]
        fm_total: FMVec_1 = jnp.concatenate([fm.TotalFM.Forces_b + f_gravity, fm.TotalFM.Moments_b], axis=0)
        assert fm_total.shape == (6, 1)
        return fm_total

    def get_inertial(self, state13: AircraftStateVec, fm: FM) -> GUAMState:
        fm_total = self.get_fm_with_gravity(state13, fm)
        state_deriv: AircraftStateVec = self.state_deriv(fm_total, state13)

        # The two things below aren't used, so should be pruned away after jit.
        inertial_data = self._get_inertial_data(state13, state_deriv)
        return inertial_data

    def get_eom_from_inertial(self, inertial_data: GUAMState, env: EnvData) -> EOM:
        # world_rel = get_world_rel(inertial_data, env)
        # air_rel = get_air_rel(world_rel, env)
        world_rel, air_rel = get_world_air_rel_simple(inertial_data, env)
        eom = EOM(inertial_data, world_rel, air_rel)
        return eom

    def get_eom(self, state13: AircraftStateVec, fm: FM, env: EnvData) -> EOM:
        inertial_data = self.get_inertial(state13, fm)
        return self.get_eom_from_inertial(inertial_data, env)

    @ft.partial(jax.jit, static_argnums=0)
    def get_sensor_aeroprop_altmsl(
        self, state13: AircraftStateVec
    ) -> tuple[SensorNoAcc, AeroPropBodyData, FloatScalar]:
        state = AircraftState.from_state13(state13)
        Vel_bEb = state.vel
        Omeg_BIb = state.omega
        Pos_bii = state.pos
        Q_i2b = state.quat

        Q_b2i = quaternion_inverse(Q_i2b)
        Vel_bIi = quaternion_vector_transformation(Q_b2i, Vel_bEb).reshape((3, 1))

        phi, theta, psi = nquaternion_to_Euler(Q_i2b)
        euler = EulerAngles(phi=phi, theta=theta, psi=psi)

        Q_h2b, AltMSL = get_Q2hb_AltMSL(Vel_bIi, Omeg_BIb, Pos_bii, Q_i2b)

        sensor = SensorNoAcc(Vel_bIi, Omeg_BIb, Pos_bii, Q_i2b, euler)
        aeroprop_body_data = AeroPropBodyData(Vel_bIi, Omeg_BIb, Q_i2b, Q_h2b)

        return sensor, aeroprop_body_data, AltMSL

    @ft.partial(jax.jit, static_argnums=0)
    def state_deriv(self, fm: FMVec_1, state13: AircraftStateVec) -> AircraftStateVec:
        assert fm.shape == (6, 1), state13.shape == (13,)
        forces = fm[:3, :]
        moments = fm[3:, :]
        state_split = AircraftState.from_state13(state13)

        omega_BIb = state_split.omega
        vel_beb = state_split.vel
        q_i2b = state_split.quat

        ddt_vb: Vec3_1 = forces / self.MASS - jnp.cross(omega_BIb, vel_beb, axis=0)
        assert ddt_vb.shape == (3, 1)

        ddt_pqr = jnp.dot(self.INVINERTIA, moments - jnp.cross(jnp.dot(self.INERTIA, omega_BIb), omega_BIb, axis=0))
        assert ddt_pqr.shape == (3, 1)

        q_f = quaternion_inverse(q_i2b)
        ddt_posi = quaternion_vector_transformation2(q_f, vel_beb).reshape((3, -1))

        input = jnp.concatenate([omega_BIb, q_i2b * 0.5])
        e0dt = -input[4] * input[0] - input[5] * input[1] - input[6] * input[2]
        eXdt = input[3] * input[0] - input[6] * input[1] + input[5] * input[2]
        eYdt = input[6] * input[0] + input[3] * input[1] - input[4] * input[2]
        eZdt = -input[5] * input[0] + input[4] * input[1] + input[3] * input[2]
        QDt_i2b = jnp.concatenate([e0dt, eXdt, eYdt, eZdt]).reshape((4, -1))
        stabilization_gain = (jnp.sum(q_i2b) - 1.0) * 0.1
        QDt_i2b = QDt_i2b - q_i2b * stabilization_gain

        # return ddt_vb, ddt_pqr, ddt_posi, QDt_i2b
        state13 = jnp.concatenate([ddt_vb, ddt_pqr, ddt_posi, QDt_i2b]).reshape(13)
        assert state13.shape == (13,)
        return state13

    def step_const_fm(self, fm: FM, state13: AircraftStateVec, dt: float) -> AircraftStateVec:
        """Temporary step function that mimics the way it is done in vehicle_eom_simp.py"""

        def deriv_fn_wrapped(t, state):
            return self.state_deriv(fm_total, state)

        fm_total = self.get_fm_with_gravity(state13, fm)
        sol = solve_ivp(deriv_fn_wrapped, [0.0, dt], state13, t_eval=[dt], method=BS5)
        assert sol.y.shape == (13, 1)
        state13_new = sol.y.squeeze(-1)
        assert state13_new.shape == (13,)
        return state13_new

    def _get_inertial_data(self, state13: AircraftStateVec, deriv: AircraftStateVec) -> GUAMState:
        state, deriv = AircraftState.from_state13(state13), AircraftState.from_state13(deriv)

        Vel_bEb = state.VelDtB_bEb
        Omeg_BIb = state.OmgDtI_BIb
        Pos_bii = state.PosDtI_bii
        Q_i2b = state.QDt_i2b
        Q_b2i = quaternion_inverse(Q_i2b)
        Vel_bIi = quaternion_vector_transformation(Q_b2i, Vel_bEb).reshape((3, 1))

        VelDtB_bEb = deriv.VelDtB_bEb
        OmgDtI_BIb = deriv.OmgDtI_BIb
        Accel_bIb = jnp.cross(Omeg_BIb, Vel_bEb, axis=0) + VelDtB_bEb
        Accel_bIi = quaternion_vector_transformation(Q_b2i, Accel_bIb).reshape((3, 1))
        Asensed_bIb = Accel_bIb - quaternion_vector_transformation(
            Q_i2b, jnp.array([[0], [0], [32.174048556430442]])
        ).reshape((3, 1))

        return GUAMState(
            pos_bii=Pos_bii,
            vel_bIi=Vel_bIi,
            accel_bii=Accel_bIi,
            asensed_bIb=Asensed_bIb,
            Q_i2b=Q_i2b,
            Omega_BIb=Omeg_BIb,
            OmegaDtl_BIb=OmgDtI_BIb,
        )

    def _get_gravity_force(self, q_i2b: Quat_1) -> Vec3:
        MASS = 181.78924904000002
        GRAVITY = 32.174048556430442
        v = np.array([0, 0, MASS * GRAVITY])
        f_gravity = quaternion_vector_transformation(q_i2b, v)
        assert f_gravity.shape == (3,)
        return f_gravity
