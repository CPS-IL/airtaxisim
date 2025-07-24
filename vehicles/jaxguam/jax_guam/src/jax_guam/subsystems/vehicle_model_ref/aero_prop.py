import pdb

import ipdb
import jax.numpy as jnp

from jax_guam.classes.TiltWingClass import TiltWingClass
from jax_guam.classes.init_tiltwing import init_tiltwing
from jax_guam.guam_types import EOM, FM, AeroFM, EnvData, PropAct, PropFM, SurfAct, TotalFM
from jax_guam.utils.functions import (
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiplication,
    quaternion_to_DCM,
    quaternion_vector_transformation,
    single_axis_quaternion,
)
from jax_guam.utils.jax_types import Mat31, Vec3_1, Vec4, Vec5, Vec9, Vec9_1


class AeroProp:
    def __init__(self):
        self.tiltwing = init_tiltwing()

    def aero_prop(self, prop_act: PropAct, surf_act: SurfAct, env: EnvData, eom: EOM) -> FM:
        USE_POLY = False
        if USE_POLY:
            return self.aero_prop_poly(prop_act, surf_act, env, eom)
        else:
            return self.aero_prop_sfn(prop_act, surf_act, env, eom)

    def aero_prop_poly(prop_act, surf_act, env, eom):
        return

    def aero_prop_sfn(self, prop_act: PropAct, surf_act: SurfAct, env: EnvData, eom: EOM) -> FM:
        eng_speed, h_b, hdot_b = self.prop_force_moments(prop_act, env)
        surf_pos = surf_act.CtrlSurfPos
        surfaces = self.map_inputs(surf_pos)
        vel_body, ang_body = self.combine_body_rates_with_winds(eom, env)

        # compute_derivatives = 0  # Note since Switches.AeroPropDeriv is 0
        compute_derivatives = False

        total_f_m, aero_total_f_m, prop_f_m, prop_t_t, fm_x, fm_u = self.lift_cruise_forces_moments(
            env.Atmosphere.Density, vel_body, ang_body, eng_speed, surfaces, compute_derivatives
        )
        Forces_b = total_f_m[:3]
        Moments_b = total_f_m[3:]
        Aero_Fb = aero_total_f_m[:3]
        Aero_Mb = aero_total_f_m[3:]
        Prop_Fb = prop_f_m[:3]
        Prop_Mb = prop_f_m[3:]
        Prop_Q = prop_t_t[0]
        Prop_T = prop_t_t[1]
        # import pdb pdb.set_trace()
        totalFM = TotalFM(Forces_b=Forces_b, Moments_b=Moments_b, H_b=h_b, Hdot_b=hdot_b)
        aeroFM = AeroFM(Aero_Fb=Aero_Fb, Aero_Mb=Aero_Mb)
        propFM = PropFM(Prop_Fb=Prop_Fb, Prop_Mb=Prop_Mb, Prop_T=Prop_T[0], Prop_Q=Prop_Q[0])
        return FM(TotalFM=totalFM, AeroFM=aeroFM, PropFM=propFM)

    def prop_force_moments(self, prop_act: PropAct, env: EnvData) -> tuple[Vec9, Vec3_1, Vec3_1]:
        eng_speed = prop_act.EngSpeed
        eng_accel = prop_act.EngAccel

        # total angulalr momentum
        Ip = jnp.array([13.4860, 13.4860, 13.4860, 13.4860, 13.4860, 13.4860, 13.4860, 13.4860, 17.4860])
        p_rot_axis_e = jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, -0.1392, 0.1392, 0, 0, -0.1392, 0.1392, 0, 0],
                [-1, -0.9903, -0.9903, -1, -1, -0.9903, -0.9903, -1, 0],
            ]
        )
        prop_dir = jnp.array([-1, 1, -1, 1, 1, -1, 1, -1, 1])
        # import pdb pdb.set_trace()
        H_h = (eng_speed.reshape(9) * Ip * prop_dir).reshape((9, 1))
        Hdot_h = (eng_accel.reshape(9) * Ip * prop_dir).reshape((9, 1))
        H_b = p_rot_axis_e.dot(H_h)
        Hdot_b = p_rot_axis_e.dot(Hdot_h)

        assert H_b.shape == (3, 1) and Hdot_b.shape == (3, 1)
        return eng_speed, H_b, Hdot_b

    def map_inputs(self, surf_pos: Vec5) -> Vec4:
        left_flaperon = surf_pos[0]
        right_flaperon = surf_pos[1]
        le = surf_pos[2]
        re = surf_pos[3]
        rudder = surf_pos[4]

        flap = (left_flaperon + right_flaperon) * 0.5
        aileron = right_flaperon - flap
        elevator = (le + re) * 0.5
        return jnp.array([aileron, flap, elevator, rudder])

    def combine_body_rates_with_winds(self, eom: EOM, env: EnvData) -> tuple[Vec3_1, Vec3_1]:
        vel_bIi = eom.InertialData.vel_bIi
        omeg_bIb = eom.InertialData.Omega_BIb
        q_i2b = eom.InertialData.Q_i2b
        q_h2b = eom.WorldRelativeData.Q_h2b
        vel_bIb = quaternion_vector_transformation(q_i2b, vel_bIi).reshape((3, 1))

        vel_whh = env.Wind.Vel_wHh
        vel_tbb = env.Turbulence.Vel_tBb
        omeg_tbb = env.Turbulence.Vel_tBb
        vel_wind_body = quaternion_vector_transformation(q_h2b, vel_whh).reshape((3, 1))
        angrates_body = omeg_tbb + omeg_bIb
        vel_body = vel_bIb + vel_wind_body + vel_tbb

        assert vel_body.shape == (3, 1) and angrates_body.shape == (3, 1)
        return vel_body, angrates_body

    def lift_cruise_forces_moments(
        self, rho: float, V_b: Vec3_1, om_b: Vec3_1, om_prop: Vec9_1, surf: Vec4, ders: bool
    ):
        assert isinstance(ders, bool)
        self.tiltwing.set_om_p(om_prop)  # Extra props
        self.tiltwing.set_del_a(surf[0])  # Aileron
        self.tiltwing.set_del_f(surf[1])  # Flap
        self.tiltwing.set_del_e(surf[2])  # Elevator
        self.tiltwing.set_del_r(surf[3])  # Rudder
        # Note: Hardcoded to zero!
        self.tiltwing.set_i_w(0)
        self.tiltwing.set_i_t(0)

        self.tiltwing: TiltWingClass = self.tiltwing.aero(rho, V_b, om_b, ders)

        Fb = self.tiltwing.total_Fb()
        Mb = self.tiltwing.total_Mb()
        FM = jnp.concatenate((Fb, Mb))

        # Note: All these derivative terms are hardcoded to zero.
        F_x = jnp.array([self.tiltwing.Fx_x, self.tiltwing.Fy_x, self.tiltwing.Fz_x])
        M_x = jnp.array([self.tiltwing.Mx_x, self.tiltwing.My_x, self.tiltwing.Mz_x])
        FM_x = jnp.concatenate((F_x, M_x))
        F_u = jnp.array([self.tiltwing.Fx_u, self.tiltwing.Fy_u, self.tiltwing.Fz_u])
        M_u = jnp.array([self.tiltwing.Mx_u, self.tiltwing.My_u, self.tiltwing.Mz_u])
        FM_u = jnp.concatenate((F_u, M_u))

        FM_aero = jnp.concatenate((self.tiltwing.aero_Fb(), self.tiltwing.aero_Mb()))
        FM_prop = jnp.concatenate((self.tiltwing.prop_Fb(), self.tiltwing.prop_Mb()))

        Prop = jnp.array([self.tiltwing.get_Qp(), self.tiltwing.get_Tp()])  # Torque, thrust

        return FM, FM_aero, FM_prop, Prop, FM_x, FM_u
