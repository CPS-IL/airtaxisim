import functools as ft
import pdb

import ipdb
import jax.numpy as jnp
import numpy as np
from extensisq import BS5
from loguru import logger
from scipy.integrate import solve_ivp

from jax_guam.guam_types import *
from jax_guam.utils.functions import (
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiplication,
    quaternion_to_DCM,
    quaternion_vector_transformation,
    quaternion_vector_transformation2,
    single_axis_quaternion,
)
from jax_guam.utils.jax_types import FloatScalar, FMVec_1, Force, Mat3, Moment, Vec3_1, Vec6, Vec6_1, Vec13


class VehicleEOMRef:
    # forcesmoments: Vec6_1

    def __init__(self):
        initial_state13: AircraftStateVec = jnp.array([-0.00069017, 0, -8, 0, 0, 0, 0, 0, 0, 1, 0, -4.3136e-05, 0])
        self.state13 = initial_state13[:, None]

        # self.Pos_bii = jnp.zeros((3, 1))
        # self.Vel_bIi = jnp.zeros((3, 1))
        # self.Accel_bIi = jnp.zeros((3, 1))
        # self.Asensed_bIb = jnp.zeros((3, 1))
        # self.Q_i2b = jnp.zeros((4, 1))
        # self.Omeg_BIb = jnp.zeros((3, 1))
        # self.OmgDtI_BIb = jnp.zeros((3, 1))

        # Note: EOM.InertialData is GUAMState.
        # self.state = GUAMState(
        #     pos_bii=self.Pos_bii,
        #     vel_bIi=self.Vel_bIi,
        #     accel_bii=self.Accel_bIi,
        #     asensed_bIb=self.Asensed_bIb,
        #     Q_i2b=self.Q_i2b,
        #     Omega_BIb=self.Omeg_BIb,
        #     OmegaDtl_BIb=self.OmgDtI_BIb,
        # )

        self.derivs = None
        # self.EOM = EOM(
        #                 InertialData=self.inertial_data,
        #                 WorldRelativeData=self.world_relative_d,
        #                 AirRelativeData=self.air_relative_d
        #             )

        self.MASS = 181.78924904000002
        self.INERTIA = np.array([[13052, 58, -2969], [58, 16661, -986], [-2969, -986, 24735]])

    # def vehicle_eom(self, env: Env) -> EOM:
    #     inertial_data = self.equ_motion_init()
    #     world_relative_d = self.world_relative_data(inertial_data, env)
    #     air_relative_d = self.air_relative_data(world_relative_d, env)
    #     return EOM(InertialData=inertial_data, WorldRelativeData=world_relative_d, AirRelativeData=air_relative_d)

    def vehicle_eom_init(self, derivs) -> EOM:
        # Note: This is called BEFORE equ_motion. We need derivs for Omega_BIb since sensors needs it and it is used.
        if derivs is None:
            derivs = self.derivs

        inertial_data = self.equ_motion_init(derivs)
        world_relative_d = self.world_relative_data_init(inertial_data)
        self.EOM = EOM(InertialData=inertial_data, WorldRelativeData=world_relative_d, AirRelativeData=None)
        return self.EOM

    def equ_motion_init(self, derivs) -> GUAMState:
        # Note: Updates internal_state by pulling from state13.
        # internal_state = AircraftState(
        #     VelDtB_bEb=self.state13[:3],
        #     OmgDtI_BIb=self.state13[3:6],
        #     PosDtI_bii=self.state13[6:9],
        #     QDt_i2b=self.state13[9:13],
        # ).verify()
        internal_state = AircraftState.from_state13(self.state13)
        state = self.create_inertial_bus(internal_state, derivs)
        return state

    def equ_motion(self, time, fm: FM, env: EnvData):
        # logger.info("FM: {}, {}".format(fm.TotalFM.Forces_b.flatten(), fm.TotalFM.Forces_b.flatten()))

        forces = fm.TotalFM.Forces_b
        moments = fm.TotalFM.Moments_b

        # Note: EOM.InertialData is GUAMState.
        # Note: Since we are only using Q_i2b, which is pulled from QDt_i2b, which is pulled from state13, it is sufficient to just take from state13.
        # state = self.state
        state = self.EOM.InertialData

        add = self.gravity(state).reshape((6, 1))
        forcesmoments: Vec6_1 = jnp.concatenate([forces, moments]) + add
        assert forcesmoments.shape == (6, 1)

        # ddt_vb, ddt_pqr, ddt_posi, QDt_i2b = self.aircraft_6dof(forcesmoments, MASS, INERTIA, self.state13)
        # self.derivs = AircraftState(VelDtB_bEb = ddt_vb,OmgDtI_BIb = ddt_pqr, PosDtI_bii = ddt_posi, QDt_i2b = QDt_i2b)
        # self.state = self.create_inertial_bus(self.internal_state, self.derivs)

        # state13 = jnp.concatenate([ddt_vb, ddt_pqr, ddt_posi, QDt_i2b])
        # # TODO: integral ddt_vb, ddt_pqr, ddt_posi, QDt_i2b = ...

        state13 = self.state13.squeeze(-1)

        state13_deriv = self.aircraft_6dof_integral(forcesmoments, time, state13).reshape((13, 1))
        derivs = AircraftState.createDeriv(
            VelDtB_bEb=state13_deriv[:3],
            OmgDtI_BIb=state13_deriv[3:6],
            PosDtI_bii=state13_deriv[6:9],
            QDt_i2b=state13_deriv[9:],
        ).verify()
        # Note: As noted above, sensors needs derivs.
        self.derivs = derivs
        # Note: This here uses the same internal_state, but just updating the derivs using the new forcesmoments?
        internal_state = AircraftState.createDeriv(
            VelDtB_bEb=self.state13[:3],
            OmgDtI_BIb=self.state13[3:6],
            PosDtI_bii=self.state13[6:9],
            QDt_i2b=self.state13[9:13],
        ).verify()
        state = self.create_inertial_bus(internal_state, derivs)

        inertial_data = state
        world_relative_d = self.world_relative_data(inertial_data, env)
        air_relative_d = self.air_relative_data(world_relative_d, env)

        # Note: The only consumer of this EOM is the environment param update.
        self.EOM = EOM(InertialData=inertial_data, WorldRelativeData=world_relative_d, AirRelativeData=air_relative_d)

        #############################################################
        # Integration.
        # state13 = self.initial_state13.reshape(13)
        deriv_fn = ft.partial(self.aircraft_6dof_integral, forcesmoments)
        sol = solve_ivp(deriv_fn, [time, time + 0.005], state13, t_eval=[time + 0.005], method=BS5)
        # (13, n_points = 1) = (13, 1)
        self.state13 = sol.y
        assert self.state13.shape == (13, 1)
        ##################################################################
        return self.EOM
        # pdb.set_trace()

    def world_relative_data_init(self, inertial_data: GUAMState) -> WorldRel:
        pos_bii = inertial_data.pos_bii
        vel_bIi = inertial_data.vel_bIi
        accel_bii = inertial_data.accel_bii
        Omega_BIb = inertial_data.Omega_BIb
        Q_i2b = inertial_data.Q_i2b
        Omeg_EIi = jnp.array([[0], [0], [0]])
        OmegDtI_EIi = jnp.array([[0], [0], [0]])
        Pos_bei, Vel_bEi, Omeg_BEb = self.ECEF_init(pos_bii, vel_bIi, Omega_BIb, Q_i2b, Omeg_EIi, OmegDtI_EIi)
        Pos_bee, Vel_bEh, Omeg_BHb, Omeg_HEh, Q_i2h, Q_h2b, Lon, LatGeod, AltGeod = self.local_ned_flattened_init(
            Pos_bei, Vel_bEi, Omeg_BEb, Q_i2b
        )
        AltMSL = AltGeod
        AltAGL = AltMSL
        Vel_bEb = quaternion_vector_transformation(Q_i2b, Vel_bEi).reshape((3, 1))
        # VelDtE_bEb = quaternion_vector_transformation(Q_i2b, VelDtE_bEi).reshape((3,1))
        # chidot, gammadot = self.flight_path_angle_rates(VelDtH_bEh, Vel_bEh)
        chi, gamma = self.flight_path_angles(Vel_bEh)
        phi, theta, psi = self.local_horizontal_to_321_euler(Q_h2b)
        euler = EulerAngles(phi=float(phi), theta=float(theta), psi=float(psi))
        Pos_beh_topo = self.topodetic_potision(Pos_bee)
        latLonAlt = LatLonAlt(LatGeod=LatGeod, Lon=Lon, AltGeod=AltGeod)
        return WorldRel(
            Pos_bee=Pos_bee,
            Pos_beh_topo=Pos_beh_topo,
            LatLonAlt=latLonAlt,
            AltMSL=AltMSL,
            AltAGL=AltAGL,
            AltPresMSL=None,
            Euler=euler,
            Q_h2b=Q_h2b,
            Q_i2h=Q_i2h,
            gamma=gamma,
            chi=chi,
            Omeg_BEb=Omeg_BEb,
            Omeg_BHb=Omeg_BHb,
            Vel_bEb=Vel_bEb,
            Vel_bEh=Vel_bEh,
            Omeg_HEh=Omeg_HEh,
            VelDtE_bEb=None,
            VelDtH_bEh=None,
            gammadot=None,
            chidot=None,
        )

    def world_relative_data(self, inertial_data: GUAMState, env: EnvData) -> WorldRel:
        pos_bii = inertial_data.pos_bii
        vel_bIi = inertial_data.vel_bIi
        accel_bii = inertial_data.accel_bii
        Omega_BIb = inertial_data.Omega_BIb
        Q_i2b = inertial_data.Q_i2b
        Omeg_EIi = jnp.array([[0], [0], [0]])
        OmegDtI_EIi = jnp.array([[0], [0], [0]])
        Pos_bei, Vel_bEi, VelDtE_bEi, Omeg_BEb = self.ECEF(
            pos_bii, vel_bIi, accel_bii, Omega_BIb, Q_i2b, Omeg_EIi, OmegDtI_EIi
        )
        (
            Pos_bee,
            Vel_bEh,
            VelDtH_bEh,
            Omeg_BHb,
            Omeg_HEh,
            Q_i2h,
            Q_h2b,
            Lon,
            LatGeod,
            AltGeod,
        ) = self.local_ned_flattened(Pos_bei, Vel_bEi, VelDtE_bEi, Omeg_BEb, Q_i2b)
        AltMSL = AltGeod
        AltAGL = AltMSL
        AltPress = self.pressure_altitude(env.Atmosphere.Pressure)
        Vel_bEb = quaternion_vector_transformation(Q_i2b, Vel_bEi).reshape((3, 1))
        VelDtE_bEb = quaternion_vector_transformation(Q_i2b, VelDtE_bEi).reshape((3, 1))
        chidot, gammadot = self.flight_path_angle_rates(VelDtH_bEh, Vel_bEh)
        chi, gamma = self.flight_path_angles(Vel_bEh)
        phi, theta, psi = self.local_horizontal_to_321_euler(Q_h2b)
        euler = EulerAngles(phi=float(phi), theta=float(theta), psi=float(psi))
        Pos_beh_topo = self.topodetic_potision(Pos_bee)
        latLonAlt = LatLonAlt(LatGeod=LatGeod, Lon=Lon, AltGeod=AltGeod)
        return WorldRel(
            Pos_bee=Pos_bee,
            Pos_beh_topo=Pos_beh_topo,
            LatLonAlt=latLonAlt,
            AltMSL=AltMSL,
            AltAGL=AltAGL,
            AltPresMSL=AltPress,
            Euler=euler,
            Q_h2b=Q_h2b,
            Q_i2h=Q_i2h,
            gamma=gamma,
            chi=chi,
            Omeg_BEb=Omeg_BEb,
            Omeg_BHb=Omeg_BHb,
            Vel_bEb=Vel_bEb,
            Vel_bEh=Vel_bEh,
            Omeg_HEh=Omeg_HEh,
            VelDtE_bEb=VelDtE_bEb,
            VelDtH_bEh=VelDtH_bEh,
            gammadot=gammadot,
            chidot=chidot,
        )

    def air_relative_data(self, world_rel: WorldRel, env: EnvData) -> AirRel:
        Density = env.Atmosphere.Density
        SpeedSound = env.Atmosphere.SpeedOfSound
        Static_Pres = env.Atmosphere.Pressure
        Vel_tBb = env.Turbulence.Vel_tBb
        VelDtB_tBb = env.Turbulence.VelDtB_tBb
        Omeg_TBb = env.Turbulence.Omeg_TBb
        Vel_wHh = env.Wind.Vel_wHh
        Vel_DtH_wHh = env.Wind.Vel_DtH_wHh
        Q_h2b = world_rel.Q_h2b
        Vel_bEb = world_rel.Vel_bEb
        VelDtE_bEb = world_rel.VelDtE_bEb
        Omeg_BEb = world_rel.Omeg_BEb
        Omeg_BHb = world_rel.Omeg_BHb
        Vel_wEb = quaternion_vector_transformation(Q_h2b, Vel_wHh).reshape((3, 1))
        VelDtH_wEb = quaternion_vector_transformation(Q_h2b, Vel_DtH_wHh).reshape((3, 1))
        Vel_bWb, VelDtB_bWb, Omeg_BWb = self.air_relative(
            Vel_tBb, VelDtB_tBb, Omeg_TBb, Vel_wEb, VelDtH_wEb, Vel_bEb, VelDtE_bEb, Omeg_BEb, Omeg_BHb
        )
        VelDtH_bVh, Vel_bVh = self.air_real_in_ned_coordinates(Vel_bWb, VelDtB_bWb, Q_h2b, Omeg_BWb)

        Vtot = jnp.sqrt(jnp.sum(Vel_bWb**2))
        Veas = Vtot * (jnp.sqrt(Density / 0.0023768907688269184))
        Mach = Vtot / SpeedSound
        qbar = 0.5 * Density * (Vtot**2)
        qc = self.supersonic(Mach, Static_Pres) if Mach > 1 else self.subsonic(Mach, Static_Pres)
        Vcas = self.calibrated_airspeed(qc)
        alpha = jnp.arctan2(Vel_bWb[2], Vel_bWb[0])
        beta = jnp.arctan2(Vel_bWb[1], jnp.sqrt(jnp.sum(jnp.array([Vel_bWb[0], Vel_bWb[2]]) ** 2)))
        chi, gamma, mu = self.air_relative_chi_gamma_mu(Q_h2b, beta, alpha)
        alphadot, betadot, Vtotdot = self.wind_angle_derivative(Vel_bWb, VelDtB_bWb)
        chidot, gammadot = self.flight_path_angle_rates(VelDtH_bVh, Vel_bVh)
        mudot = self.velocity_vector_bank_angle_rate(alpha, beta, gamma, chidot, alphadot, Omeg_BHb)
        phiAero = jnp.arctan2(Vel_bWb[1], Vel_bWb[2])
        alphaTotal = jnp.arctan2(jnp.sqrt(jnp.sum(Vel_bWb[1:] ** 2)), Vel_bWb[0])
        Q_h2v, Omeg_VHb = self.velocity_vector_altitude_and_regular_rate(chi, gamma, chidot, gammadot, Q_h2b)
        return AirRel(
            Vel_bWb=Vel_bWb,
            VelDtB_bWb=VelDtB_bWb,
            Omeg_BWb=Omeg_BWb,
            Vtot=Vtot,
            Veas=Veas,
            Vcas=Vcas,
            Mach=Mach,
            qbar=qbar,
            qc=qc,
            alpha=alpha,
            beta=beta,
            mu=mu,
            gamma=gamma,
            chi=chi,
            Vtotdot=Vtotdot,
            alphadot=alphadot,
            betadot=betadot,
            mudot=mudot,
            gammadot=gammadot,
            chidot=chidot,
            alphaTotal=alphaTotal,
            phiAero=phiAero,
            Q_h2v=Q_h2v,
            Omeg_VHb=Omeg_VHb,
        )

    ################### Helper function for air_relative_data()
    def air_relative(self, Vel_tBb, VelDtB_tBb, Omeg_TBb, Vel_wEb, VelDtH_wEb, Vel_bEb, VelDtE_bEb, Omeg_BEb, Omeg_BHb):
        # pdb.set_trace()
        Vel_bWb = Vel_bEb - Vel_wEb - Vel_tBb
        VelDtB_bWb = (
            (VelDtE_bEb - jnp.cross(Omeg_BEb, Vel_bEb, axis=0))
            - (VelDtH_wEb - jnp.cross(Omeg_BHb, Vel_wEb, axis=0))
            - VelDtB_tBb
        )
        Omeg_BWb = Omeg_BHb - Omeg_TBb
        return Vel_bWb, VelDtB_bWb, Omeg_BWb

    def air_real_in_ned_coordinates(self, Vel_bWb, VelDtB_bWb, Q_h2b, Omeg_BWb):
        q = quaternion_conjugate(Q_h2b)
        # import pdb; pdb.set_trace()
        v_1 = VelDtB_bWb + jnp.cross(Omeg_BWb, Vel_bWb, axis=0)
        VelDtH_bVh = quaternion_vector_transformation(q, v_1)
        Vel_bVh = quaternion_vector_transformation(q, Vel_bWb)
        return VelDtH_bVh, Vel_bVh

    def supersonic(self, Mach, Static_Pres):
        Mach = Mach if Mach >= 1 else 1
        gamma = 1.4
        x_1 = Mach**2 * (gamma + 1) / 2
        u_1 = gamma / (gamma - 1)
        x_2 = Mach**2 * 2 * gamma / (gamma + 1) - (gamma - 1) / (gamma + 1)
        u_2 = 1 / (gamma - 1)
        qc = ((x_1 ** (u_1)) * (x_2 ** (u_2)) - 1) * Static_Pres
        return qc

    def subsonic(self, Mach, Static_Pres):
        gamma = 1.4
        x_1 = Mach**2 * (gamma - 1) / 2 + 1
        u_1 = gamma / (gamma - 1)
        qc = ((x_1 ** (u_1)) - 1) * Static_Pres
        return qc

    def calibrated_airspeed(self, qc):
        gamma = 1.4
        Pref = 2116.2166236739372
        rhoref = 0.0023768907688269184
        mul_1 = (qc / Pref + 1) ** ((gamma - 1) / gamma) - 1
        Vcas = jnp.sqrt(mul_1 * Pref / rhoref * (2 * gamma / (gamma - 1)))
        return Vcas

    def air_relative_chi_gamma_mu(self, Q_h2b, beta, alpha):
        q2 = quaternion_conjugate(
            quaternion_multiplication(single_axis_quaternion(-beta, 2), single_axis_quaternion(alpha, 1))
        )
        # pdb.set_trace()
        dcm = quaternion_to_DCM(quaternion_multiplication(Q_h2b, q2))
        chi = jnp.arctan2(dcm[0][1], dcm[0][0])
        gamma = jnp.arcsin(-dcm[0][2])
        mu = jnp.arctan2(dcm[1][2], dcm[2][2])
        return chi, gamma, mu

    def wind_angle_derivative(self, Vel_bVb, VelDtB_bVb):
        mult = Vel_bVb[0] * VelDtB_bVb[2] - Vel_bVb[2] * VelDtB_bVb[0]  # Note
        condition = jnp.sum(Vel_bVb[[0, 2], :] ** 2)
        div = condition if condition > 0 else 1
        alphadot = mult / div if div > 0 else 0
        div2 = jnp.sqrt(div) if jnp.sqrt(div) > 0 else 1
        div3 = jnp.sqrt(jnp.sum(Vel_bVb**2)) if jnp.sqrt(jnp.sum(Vel_bVb**2)) > 0 else 1
        mult3 = jnp.dot(Vel_bVb.T, VelDtB_bVb) / div3
        mult2 = VelDtB_bVb[1] - VelDtB_bVb[1] * mult3 * div3
        betadot = mult2 / div2 if condition > 0 else 0
        Vinfdot = mult3 if div3 > 0 else 0
        return alphadot, betadot, Vinfdot

    def flight_path_angle_rates(self, VelDtH_bEh, Vel_bEh):
        sum_1 = jnp.dot(VelDtH_bEh[:2].T, Vel_bEh[:2]) * Vel_bEh[2]  # Note name divide but seems multiply
        minus_1 = VelDtH_bEh[2] * jnp.dot(Vel_bEh[:2].T, Vel_bEh[:2])
        mult_1 = sum_1 - minus_1
        condition = jnp.sqrt(jnp.sum(Vel_bEh[:2] ** 2)) * jnp.dot(Vel_bEh.T, Vel_bEh)  # Note
        div_1 = condition if condition != 0 else 1
        gammadot = mult_1 / div_1 if condition != 0 else 0

        mult_2 = Vel_bEh[0] * VelDtH_bEh[1] - Vel_bEh[1] * VelDtH_bEh[2]  # Note
        condition_2 = jnp.dot(Vel_bEh[:2].T, Vel_bEh[:2])
        div_2 = condition_2 if condition_2 != 0 else 1
        chidot = mult_2 / div_2 if condition_2 != 0 else 0
        return chidot, gammadot

    def velocity_vector_bank_angle_rate(self, alpha, beta, gamma, chidot, alphadot, Omeg_BHb):
        sa = jnp.sin(alpha)
        ca = jnp.cos(alpha)
        sb = jnp.sin(beta)
        cb = jnp.cos(beta)
        plus1 = jnp.sin(gamma) * chidot
        minus1 = sb * alphadot
        plus2 = cb * ca * Omeg_BHb[0][0]
        plus3 = sb * Omeg_BHb[1][0]
        plus4 = cb * sa * Omeg_BHb[2][0]
        return plus1 - minus1 + plus2 + plus3 + plus4

    def velocity_vector_altitude_and_regular_rate(self, chi, gamma, chidot, gammadot, Q_h2b):
        Q_h2v = quaternion_multiplication(single_axis_quaternion(chi, 2), single_axis_quaternion(gamma, 1))
        Q_v2b = quaternion_multiplication(quaternion_conjugate(Q_h2v), Q_h2b)
        Omeg_NHn = jnp.array([chidot * -jnp.sin(gamma), gammadot, chidot * jnp.cos(gamma)])
        Omeg_VHb = quaternion_vector_transformation(Q_v2b, Omeg_NHn)
        return Q_h2v, Omeg_VHb

    ################### Helper function for world_relative_data()
    def topodetic_potision(self, Pos_bee):
        C = jnp.array([[0], [0], [0]])
        Q_e2h = jnp.array([[1], [0], [0], [0]])
        return quaternion_vector_transformation(Q_e2h, Pos_bee - C)

    def local_horizontal_to_321_euler(self, Q_h2b):
        Q_b2a = quaternion_conjugate(Q_h2b)
        v_1 = quaternion_vector_transformation(Q_b2a, jnp.array([Q_b2a[0][0] + 1, 0, 0]))
        psi = float(jnp.arctan2(v_1[1], v_1[0]))
        q_0to1 = single_axis_quaternion(psi * 0.5, 2)
        # jnp.concatenate((jnp.cos(psi*0.5), jnp.sin(psi*0.5)*jnp.array([[0],[0],[1]])))
        U1_irt3_1 = quaternion_vector_transformation(q_0to1, v_1)
        theta = float(jnp.arctan2(-U1_irt3_1[2], U1_irt3_1[0]))
        q_1to2 = single_axis_quaternion(theta * 0.5, 1)
        # jnp.concatenate((jnp.cos(theta*0.5), jnp.sin(theta*0.5)*jnp.array([[0],[1],[0]])))
        v_2 = quaternion_vector_transformation(Q_b2a, jnp.array([0, Q_b2a[0][0] + 1, 0]))
        q_0to2 = quaternion_multiplication(q_0to1, q_1to2)
        U1_irt3_1 = quaternion_vector_transformation(q_0to2, v_2)
        phi = float(jnp.arctan2(U1_irt3_1[2], U1_irt3_1[1]))
        return phi, theta, psi

    def flight_path_angles(self, Vel_bEh):
        chi = jnp.arctan2(Vel_bEh[1], Vel_bEh[0])
        gamma = jnp.arctan2(-Vel_bEh[2], jnp.sqrt(jnp.sum(jnp.array([Vel_bEh[0], Vel_bEh[1]]) ** 2)))
        return chi, gamma

    def pressure_altitude(self, p):
        UP = 1
        KP = 0.00047254133200895748
        EP = 0.19201354867458559
        KA = 145442.15626892794
        # KP = 0.00047254
        # EP = 0.19201
        # KA = 145440
        UA = 1
        return (1 - (p * UP * KP) ** (EP)) * KA * UA

    def ECEF_init(self, pos_bii: Vec3_1, vel_bIi: Vec3_1, Omega_BIb: Vec3_1, Q_i2b: Quat_1, Omeg_EIi, OmegDtI_EIi):
        cross = jnp.cross(Omeg_EIi, vel_bIi, axis=0)
        Vel_bEi = vel_bIi - cross
        Omeg_BEb = Omega_BIb - quaternion_vector_transformation(Q_i2b, Omeg_EIi).reshape((3, 1))
        Pos_bei = pos_bii
        return Pos_bei, Vel_bEi, Omeg_BEb

    def ECEF(
        self,
        pos_bii: Vec3_1,
        vel_bIi: Vec3_1,
        accel_bii: Vec3_1,
        Omega_BIb: Vec3_1,
        Q_i2b: Quat_1,
        Omeg_EIi: Vec3_1,
        OmegDtI_EIi: Vec3_1,
    ) -> tuple[Vec3_1, Vec3_1, Vec3_1, Vec3_1]:
        cross = jnp.cross(Omeg_EIi, vel_bIi, axis=0)
        VelDtE_bEi = (
            accel_bii - jnp.cross(OmegDtI_EIi, pos_bii, axis=0) - cross * 2 + jnp.cross(Omeg_EIi, cross, axis=0)
        )
        Vel_bEi = vel_bIi - cross
        Omeg_BEb = Omega_BIb - quaternion_vector_transformation(Q_i2b, Omeg_EIi).reshape((3, 1))
        Pos_bei = pos_bii
        return Pos_bei, Vel_bEi, VelDtE_bEi, Omeg_BEb

    def local_ned_flattened_init(self, Pos_bei, Vel_bEi, Omeg_BEb, Q_i2b):
        Q_i2e = jnp.array([[1], [0], [0], [0]])
        Pos_bee = quaternion_vector_transformation(Q_i2e, Pos_bei).reshape((3, 1))  # 1
        Lon, LatGeod, AltGeod = self.flat_earth_to_geodetic(Pos_bee)
        CONSTANT_1 = jnp.array([[1], [0], [0], [0]])
        Q_i2h = CONSTANT_1 if CONSTANT_1[0][0] > 0 else -CONSTANT_1  # Note: not sure if correct #6
        Q_h2i = quaternion_conjugate(CONSTANT_1)
        Q_h2b = quaternion_multiplication(Q_h2i, Q_i2b)
        Omeg_HEh = jnp.array([[0], [0], [0]])  # 5
        Omeg_BHb = Omeg_BEb - quaternion_vector_transformation(Q_h2b, Omeg_HEh).reshape((3, 1))  # 4
        Q_h2b = Q_h2b if Q_h2b[0][0] > 0 else -Q_h2b  # Note: not sure if correct #7
        # VelDtE_bEh = quaternion_vector_transformation(CONSTANT_1, VelDtE_bEi).reshape((3,1))
        Vel_bEh = quaternion_vector_transformation(CONSTANT_1, Vel_bEi).reshape((3, 1))  # 2
        # VelDtH_bEh = VelDtE_bEh - jnp.cross(Omeg_HEh, Vel_bEh,axis=0) #3
        return Pos_bee, Vel_bEh, Omeg_BHb, Omeg_HEh, Q_i2h, Q_h2b, Lon, LatGeod, AltGeod

    def local_ned_flattened(self, Pos_bei, Vel_bEi, VelDtE_bEi, Omeg_BEb, Q_i2b):
        Q_i2e = jnp.array([[1], [0], [0], [0]])
        Pos_bee = quaternion_vector_transformation(Q_i2e, Pos_bei).reshape((3, 1))  # 1
        Lon, LatGeod, AltGeod = self.flat_earth_to_geodetic(Pos_bee)
        CONSTANT_1 = jnp.array([[1], [0], [0], [0]])
        Q_i2h = CONSTANT_1 if CONSTANT_1[0][0] > 0 else -CONSTANT_1  # Note: not sure if correct #6
        Q_h2i = quaternion_conjugate(CONSTANT_1)
        Q_h2b = quaternion_multiplication(Q_h2i, Q_i2b)
        Omeg_HEh = jnp.array([[0], [0], [0]])  # 5
        Omeg_BHb = Omeg_BEb - quaternion_vector_transformation(Q_h2b, Omeg_HEh).reshape((3, 1))  # 4
        Q_h2b = Q_h2b if Q_h2b[0][0] > 0 else -Q_h2b  # Note: not sure if correct #7
        VelDtE_bEh = quaternion_vector_transformation(CONSTANT_1, VelDtE_bEi).reshape((3, 1))
        Vel_bEh = quaternion_vector_transformation(CONSTANT_1, Vel_bEi).reshape((3, 1))  # 2
        VelDtH_bEh = VelDtE_bEh - jnp.cross(Omeg_HEh, Vel_bEh, axis=0)  # 3
        return Pos_bee, Vel_bEh, VelDtH_bEh, Omeg_BHb, Omeg_HEh, Q_i2h, Q_h2b, Lon, LatGeod, AltGeod

    ################### Helper function for equ_motion()
    def flat_earth_to_geodetic(self, Pos_bee):
        altitude = -Pos_bee[2]
        INITIAL_LATITUDE = 0
        INITIAL_LONGTITUDE = 0
        e = 0.081819190842621486
        a = 20926000
        N = a / jnp.sqrt((1 - (jnp.sin(INITIAL_LATITUDE) ** 2) * (e**2)))
        M = (1 - e**2) * N / (1 - (jnp.sin(INITIAL_LATITUDE) ** 2) * (e**2))

        long_in = jnp.arctan2(1, jnp.cos(INITIAL_LATITUDE) * N) * Pos_bee[1] + INITIAL_LONGTITUDE
        lat_in = jnp.arctan2(1, M) * Pos_bee[0] + INITIAL_LATITUDE

        Min = -jnp.pi
        Range = jnp.pi * 2
        u3 = jnp.mod(lat_in - Min, Range) + Min
        u2 = jnp.abs(u3)
        u1 = (jnp.pi - u2) * jnp.sign(u3)
        if u2 > jnp.pi / 2:
            latitude = u1
        else:
            latitude = u3

        u3_2 = long_in
        u2_2 = u2
        u1_2 = jnp.pi + long_in
        if u2_2 > jnp.pi / 2:
            longtitude = u1_2
        else:
            longtitude = u3_2
        return float(longtitude), float(latitude), float(altitude)

    def gravity(self, state: GUAMState) -> Vec6:
        MASS = 181.78924904000002
        GRAVITY = 32.174048556430442
        q_i2b = state.Q_i2b
        product = MASS * GRAVITY
        v = jnp.array([0, 0, product])
        v_b = quaternion_vector_transformation(q_i2b, v)
        return jnp.concatenate([v_b, jnp.array([0, 0, 0])])

    # def aircraft_6dof(self, forcesmoments: Vec6, mass: FloatScalar, inertia: Mat3, state13: AircraftState) -> AircraftState:
    #     forces = forcesmoments[:3, :]
    #     moments = forcesmoments[3:, :]
    #     invinertia = np.linalg.inv(inertia) #TODO linalg.inv
    #     omega_BIb = state13[3:6, :]
    #     vel_beb = state13[:3, :]
    #     q_i2b = state13[9:13, :]
    #     # import pdb; pdb.set_trace()
    #     ddt_vb = forces/mass - jnp.cross(omega_BIb, vel_beb,axis=0)

    #     ddt_pqr = jnp.dot(invinertia, moments - jnp.cross(jnp.dot(inertia, omega_BIb), omega_BIb,axis=0))

    #     q_f = quaternion_inverse(q_i2b)
    #     ddt_posi = quaternion_vector_transformation(q_f, vel_beb).reshape((3,1))

    #     input = jnp.concatenate([omega_BIb, q_i2b*0.5])
    #     e0dt = -input[4]*input[0] -input[5]*input[1] -input[6]*input[2]
    #     eXdt = input[3]*input[0] -input[6]*input[1] +input[5]*input[2]
    #     eYdt = input[6]*input[0] +input[3]*input[1] -input[4]*input[2]
    #     eZdt = -input[5]*input[0] +input[4]*input[1] +input[3]*input[2]
    #     QDt_i2b = jnp.concatenate([e0dt, eXdt, eYdt, eZdt]).reshape((4,1))
    #     stabilization_gain = (jnp.sum(q_i2b) - 1.0) * 0.1
    #     QDt_i2b = QDt_i2b - q_i2b * stabilization_gain

    #     # return ddt_vb, ddt_pqr, ddt_posi, QDt_i2b

    def aircraft_6dof_integral(self, forcesmoments: FMVec_1, time, state13: AircraftStateVec) -> AircraftStateVec:
        assert forcesmoments.shape == (6, 1), state13.shape == (13,)
        state13 = state13.reshape((13, -1))
        forces = forcesmoments[:3, :]
        moments = forcesmoments[3:, :]
        invinertia = np.linalg.inv(self.INERTIA)  # TODO linalg.inv
        omega_BIb = state13[3:6, :]
        vel_beb = state13[:3, :]
        q_i2b = state13[9:13, :]
        # import pdb; pdb.set_trace()
        ddt_vb: Vec3_1 = forces / self.MASS - jnp.cross(omega_BIb, vel_beb, axis=0)
        assert ddt_vb.shape == (3, 1)

        ddt_pqr = jnp.dot(invinertia, moments - jnp.cross(jnp.dot(self.INERTIA, omega_BIb), omega_BIb, axis=0))
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

    def create_inertial_bus(self, internal_state: AircraftState, derivs: AircraftState) -> GUAMState:
        Vel_bEb = internal_state.VelDtB_bEb
        Omeg_BIb = internal_state.OmgDtI_BIb
        Pos_bii = internal_state.PosDtI_bii
        Q_i2b = internal_state.QDt_i2b
        Q_b2i = quaternion_inverse(Q_i2b)
        Vel_bIi = quaternion_vector_transformation(Q_b2i, Vel_bEb).reshape((3, 1))
        if derivs is not None:
            VelDtB_bEb = derivs.VelDtB_bEb
            OmgDtI_BIb = derivs.OmgDtI_BIb
            Accel_bIb = jnp.cross(Omeg_BIb, Vel_bEb, axis=0) + VelDtB_bEb
            Accel_bIi = quaternion_vector_transformation(Q_b2i, Accel_bIb).reshape((3, 1))
            Asensed_bIb = Accel_bIb - quaternion_vector_transformation(
                Q_i2b, jnp.array([[0], [0], [32.174048556430442]])
            ).reshape((3, 1))
        else:
            Accel_bIi = np.zeros((3, 1))
            Asensed_bIb = np.zeros((3, 1))
            OmgDtI_BIb = np.zeros((3, 1))

        return GUAMState(
            pos_bii=Pos_bii,
            vel_bIi=Vel_bIi,
            accel_bii=Accel_bIi,
            asensed_bIb=Asensed_bIb,
            Q_i2b=Q_i2b,
            Omega_BIb=Omeg_BIb,
            OmegaDtl_BIb=OmgDtI_BIb,
        )
