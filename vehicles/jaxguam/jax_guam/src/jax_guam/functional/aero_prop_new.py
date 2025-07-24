import functools as ft

import jax
import jax.numpy as jnp
import numpy as np

from jax_guam.functional.init_func_tiltwing import init_func_tiltwing
from jax_guam.functional.tilt_wing import TiltwingControls
from jax_guam.guam_types import EOM, FM, EnvData, PropAct, SurfAct, TotalFM, TraditionalSurfaces, AeroFM, PropFM, \
    AeroPropBodyData
from jax_guam.utils.functions import quaternion_vector_transformation
from jax_guam.utils.jax_types import Vec3_1, Vec4, Vec5, Vec9, Vec9_1, FMVec


class FuncAeroProp:
    def __init__(self):
        self.tiltwing = init_func_tiltwing()

    @ft.partial(jax.jit, static_argnums=0)
    def aero_prop(self, prop_act: PropAct, surf_act: SurfAct, env: EnvData, eom: AeroPropBodyData) -> FM:
        USE_POLY = False
        assert USE_POLY is False
        return self.aero_prop_sfn(prop_act, surf_act, env, eom)

    def aero_prop_sfn(self, prop_act: PropAct, surf_act: SurfAct, env: EnvData, eom: AeroPropBodyData) -> FM:
        eng_speed, h_b, hdot_b = prop_force_moments(prop_act)
        surf_pos: Vec5 = surf_act.CtrlSurfPos
        surfs = map_inputs(surf_pos)
        vel_body, ang_body = combine_body_rates_with_winds(eom, env)

        compute_derivs = False

        # lift_cruise_forces_moments
        rho = env.Atmosphere.Density
        controls = TiltwingControls(eng_speed, surfs.aileron, surfs.flap, surfs.elevator, surfs.rudder, 0, 0)
        fm_total, fm_aero, fm_prop = self.lift_cruise_force_moments(rho, vel_body, ang_body, controls, compute_derivs)
        fm_total = TotalFM(fm_total[:3, None], fm_total[3:, None], h_b, hdot_b)
        return FM(fm_total, fm_aero, fm_prop)

    def lift_cruise_force_moments(
        self, rho: float, V_b: Vec3_1, om_b: Vec3_1, controls: TiltwingControls, ders: bool
    ) -> tuple[FMVec, AeroFM, PropFM]:
        assert isinstance(ders, bool) and ders is False
        return self.tiltwing.aero(rho, V_b, om_b, ders, controls)


def prop_force_moments(prop_act: PropAct) -> tuple[Vec9, Vec3_1, Vec3_1]:
    # Note: Originally had Env, but unused.
    eng_speed = prop_act.EngSpeed
    eng_accel = prop_act.EngAccel

    # total angular momentum
    Ip = np.array([13.4860, 13.4860, 13.4860, 13.4860, 13.4860, 13.4860, 13.4860, 13.4860, 17.4860])
    p_rot_axis_e = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, -0.1392, 0.1392, 0, 0, -0.1392, 0.1392, 0, 0],
            [-1, -0.9903, -0.9903, -1, -1, -0.9903, -0.9903, -1, 0],
        ]
    )
    assert p_rot_axis_e.shape == (3, 9)
    # Some props spin cw, some spin ccw.
    prop_dir = np.array([-1, 1, -1, 1, 1, -1, 1, -1, 1])
    # import pdb pdb.set_trace()
    H_h = (eng_speed.reshape(9) * Ip * prop_dir).reshape((9, 1))
    Hdot_h = (eng_accel.reshape(9) * Ip * prop_dir).reshape((9, 1))

    # (3, 9) @ (9, 1) = (3, 1)
    H_b = p_rot_axis_e @ H_h
    Hdot_b = p_rot_axis_e @ Hdot_h

    assert H_b.shape == (3, 1) and Hdot_b.shape == (3, 1)
    return eng_speed, H_b, Hdot_b


def map_inputs(surf_pos: Vec5) -> TraditionalSurfaces:
    """Map the left/right Flaperons and left/right elevators to a central one."""
    left_flaperon = surf_pos[0]
    right_flaperon = surf_pos[1]
    le = surf_pos[2]
    re = surf_pos[3]
    rudder = surf_pos[4]

    flap = (left_flaperon + right_flaperon) * 0.5
    aileron = right_flaperon - flap
    elevator = (le + re) * 0.5
    return TraditionalSurfaces(aileron, flap, elevator, rudder)


def combine_body_rates_with_winds(eom: AeroPropBodyData, env: EnvData) -> tuple[Vec3_1, Vec3_1]:
    vel_bIi = eom.vel_bIi
    omeg_bIb = eom.Omega_BIb
    q_i2b = eom.Q_i2b
    q_h2b = eom.Q_h2b
    vel_bIb = quaternion_vector_transformation(q_i2b, vel_bIi).reshape((3, 1))

    vel_whh = env.Wind.Vel_wHh
    vel_tbb = env.Turbulence.Vel_tBb
    omeg_tbb = env.Turbulence.Omeg_TBb
    vel_wind_body = quaternion_vector_transformation(q_h2b, vel_whh).reshape((3, 1))
    angrates_body = omeg_tbb + omeg_bIb
    vel_body = vel_bIb + vel_wind_body + vel_tbb

    assert vel_body.shape == (3, 1) and angrates_body.shape == (3, 1)
    return vel_body, angrates_body
