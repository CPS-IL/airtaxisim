import functools as ft
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from attrs import define

from jax_guam.data.read_data import read_data
from jax_guam.guam_types import (
    Alloc,
    Cmd,
    Control,
    Ctrl_Sys_Lat,
    Ctrl_Sys_Lon,
    CtrlSys,
    EngineControl_1,
    RefInputs,
    SensorNoAcc,
)
from jax_guam.utils.functions import (
    get_lininterp_idx,
    matrix_interpolation,
    pseudo_inverse,
    quaternion_rotz,
    quaternion_vector_transformation,
)
from jax_guam.utils.jax_types import (
    FloatScalar,
    Vec1_1,
    Vec2,
    Vec2_1,
    Vec3,
    Vec3_1,
    Vec4_1,
    Vec5_1,
    Vec6,
    Vec8,
    Vec10,
    Vec11,
    Vec12_1,
    Vec13_1,
)
from jax_guam.utils.paths import data_dir

# only implemented baseline, no TRIM BASELINE_L1, BASELINE_AGI

# State of the longitudinal system. [ u; w; q; θ] = [ fwd_vel; up_vel; pitch rate; pitch ]
XLon = Vec4_1
# "Desired state" of longitudinal system. [ u; w; 0 ] = [ fwd_vel; up_vel; zero pitch? ]
XLonCmd = Vec3_1
# State of the lateral system. [ v; p; r; ϕ ] = [ side_vel; roll rate; yaw rate; roll ]
XLat = Vec4_1
# "Desired state" of lateral system. [ v; χ̇ ] = [ side_vel; yaw rate? ]
XLatCmd = Vec3_1


class XLonLat(NamedTuple):
    Xlon: XLon
    Xlon_cmd: XLonCmd
    Xlat: XLat
    Xlat_cmd: XLatCmd


class LCControlState(NamedTuple):
    int_e_long: Vec3
    int_e_lat: Vec3

    @staticmethod
    def create():
        return LCControlState(int_e_long=np.zeros(3), int_e_lat=np.zeros(3))


class LCDerivCache(NamedTuple):
    ctrl_sys: CtrlSys
    feedback: Vec2
    Xlonlat: XLonLat


@define
class LCControlCfg:
    ref_traj_on: bool = True
    feedback_current: bool = True
    position_error: bool = False


class LCControl:
    def __init__(self, cfg: LCControlCfg = LCControlCfg()):
        self.mat = read_data(data_dir() / "trim_table.mat")

        self.REF_TRAJ_ON = cfg.ref_traj_on
        self.FEEDBACK_CURRENT = cfg.feedback_current
        self.POSITION_ERROR = cfg.position_error

        self.MAX_POS_ENGINE = np.full((9, 1), 350.0)
        self.MIN_POS_ENGINE = np.full((9, 1), -1e-3)

    @ft.partial(jax.jit, static_argnums=0)
    def get_control(
        self, state: LCControlState, sensor: SensorNoAcc, ref_inputs: RefInputs
    ) -> tuple[Control, LCDerivCache]:
        # 1: Compute "mdes" for longitudinal and lateral.
        XU0, ctrl_sys, mdes, Xlonlat = self.baseline(state, sensor, ref_inputs)

        # 2: (Ignore adaptive)

        # 3: Allocate controls
        eng_cmd, surf_cmd, feedback, u_alloc = self.pseudo_inverse_control_alloc(XU0, ctrl_sys, mdes)
        cmd = Cmd(
            EngineCmd=eng_cmd,
            EnginePwr=np.ones((9, 1)),
            CtrlSurfaceCmd=surf_cmd,
            CtrlSurfacePwr=np.ones((5, 1)),
            GearCmd=1,
        )
        control = Control(Cmd=cmd, Alloc=u_alloc)
        return control, LCDerivCache(ctrl_sys, feedback, Xlonlat)

    @ft.partial(jax.jit, static_argnums=0)
    def state_deriv(self, cache: LCDerivCache) -> LCControlState:
        return LCControlState(self.d_int_e_long(cache), self.d_int_e_lat(cache))

    def d_int_e_long(self, cache: LCDerivCache) -> FloatScalar:
        Kv = cache.ctrl_sys.lon.Kv
        F = cache.ctrl_sys.lon.F
        C = cache.ctrl_sys.lon.C
        Cv = cache.ctrl_sys.lon.Cv
        Xlon, Xlon_cmd = cache.Xlonlat.Xlon, cache.Xlonlat.Xlon_cmd
        theta_fb = cache.feedback[0]
        d_int_e_long = jnp.dot(F, Xlon_cmd) - jnp.dot(C, Xlon) + jnp.dot(Kv, (theta_fb - jnp.dot(Cv, Xlon)))
        d_int_e_long = d_int_e_long.squeeze(-1)
        assert d_int_e_long.shape == (3,)
        return d_int_e_long

    def d_int_e_lat(self, cache: LCDerivCache) -> FloatScalar:
        Kv = cache.ctrl_sys.lat.Kv
        F = cache.ctrl_sys.lat.F
        C = cache.ctrl_sys.lat.C
        Cv = cache.ctrl_sys.lat.Cv
        Xlat, Xlat_cmd = cache.Xlonlat.Xlat, cache.Xlonlat.Xlat_cmd
        phi_fb = cache.feedback[1]
        d_int_e_lat = jnp.dot(F, Xlat_cmd) - jnp.dot(C, Xlat) + jnp.dot(Kv, (phi_fb - jnp.dot(Cv, Xlat)))
        d_int_e_lat = d_int_e_lat.squeeze(-1)
        assert d_int_e_lat.shape == (3,)
        return d_int_e_lat

    def pseudo_inverse_control_alloc(self, XU0: tuple[Vec12_1, Vec13_1], ctrl_sys: CtrlSys, mdes: tuple[Vec3, Vec3]):
        X0, U0 = XU0
        mdes_lon, mdes_lat = mdes

        # Last squares thing to allocate control.
        M_lon = pseudo_inverse(ctrl_sys.lon.W, ctrl_sys.lon.B)
        M_lat = pseudo_inverse(ctrl_sys.lat.W, ctrl_sys.lat.B)
        assert M_lon.shape == (12, 3)
        assert M_lat.shape == (11, 3)

        # (12, )
        out_lon = jnp.dot(M_lon, mdes_lon)
        # (11, )
        out_lat = jnp.dot(M_lat, mdes_lat)
        assert out_lon.shape == (12,) and out_lat.shape == (11,)

        u_lon: Vec11 = out_lon[:-1]
        theta: FloatScalar = out_lon[-1]
        u_lat: Vec10 = out_lat[:-1]
        phi: FloatScalar = out_lat[-1]
        om_r_lon: Vec8 = u_lon[:8]
        om_p = u_lon[8]
        dele = u_lon[9]
        delf = u_lon[10]
        om_r_lat: Vec8 = u_lat[:8]
        dela = u_lat[8]
        delr = u_lat[9]
        om_r: Vec8 = om_r_lon + om_r_lat

        # Translate traditional aircraft commands to the lift+cruise ones.
        eng_cmd = self.engine_command(om_r, om_p, U0)
        SurfCmd = self.surface_command(dela, delf, dele, delr, U0)

        u_alloc = self.create_allocation_bus(jnp.concatenate(mdes), ctrl_sys.lon, ctrl_sys.lat, u_lon, u_lat, U0, X0)
        # pdb.set_trace()
        feedback = jnp.array([theta, phi])
        return eng_cmd, SurfCmd, feedback, u_alloc

    def engine_command(self, om_r: Vec8, om_p: FloatScalar, U0: Vec13_1) -> EngineControl_1:
        """Compute the command for the engine controlling the propellers.
        :param om_r: ω rotors
        :param om_p: ω pusher
        :param U0: Trim control, used as a bias.
        :return: The final engine command (9, ). [ rotor (8,), pusher (1,) ].
        """
        assert om_r.shape == (8,) and om_p.shape == tuple() and U0.shape == (13, 1)
        eng_bias = U0[4:, :]
        engine_cmd = eng_bias + jnp.concatenate([om_r, om_p[None]], axis=0)[:, None]
        assert engine_cmd.shape == (9, 1)
        engine_cmd = engine_cmd.clip(self.MIN_POS_ENGINE, self.MAX_POS_ENGINE)
        return engine_cmd

    def surface_command(
        self, dela: FloatScalar, delf: FloatScalar, dele: FloatScalar, delr: FloatScalar, U0: Vec13_1
    ) -> Vec5_1:
        """Convert the command from traditional controls [ aileron; flap; elevator; rudder ] to
            [ flaperon L, flaperon R, elevator L, elevator R, rudder ]
        :param dela:
        :param delf:
        :param dele:
        :param delr:
        :param U0: Trim control, used as a bias.
        :return: (5, 1)
        """
        bias1: Vec2_1 = np.array([[-1], [1]]) * U0[1] + U0[0]
        bias2: Vec2_1 = np.array([[1], [1]]) * U0[2]
        bias3: Vec1_1 = jnp.array([U0[3]])
        surfBias = jnp.concatenate((bias1, bias2, bias3))
        assert surfBias.shape == (5, 1)

        # Translate aileron and flaps to flaperons.
        alloc_flap = np.array([[1], [1], [0], [0], [0]])
        alloc_ail = np.array([[-1], [1], [0], [0], [0]])
        alloc_elev = np.array([[0], [0], [1], [1], [0]])
        alloc_rud = np.array([[0], [0], [0], [0], [1]])
        sum_1 = delf * alloc_flap + dela * alloc_ail + dele * alloc_elev + delr * alloc_rud
        assert sum_1.shape == (5, 1)

        a_max = np.ones((5, 1)) * 0.5236
        a_min = np.ones((5, 1)) * (-0.5236)
        surfCmd = jnp.clip(sum_1 + surfBias, a_min, a_max)
        assert surfCmd.shape == (5, 1)
        return surfCmd

    def baseline(self, state: LCControlState, sensor: SensorNoAcc, ref_inputs: RefInputs):
        Vel_blc, e_pos_blc, e_chi = self.convert_velocity_position_error_to_control_frame(sensor, ref_inputs)
        X0, U0, ctrl_sys = self.scheduled_parameters(ref_inputs)
        Xlon, Xlon_cmd, Xlat_cmd, Xlat = self.perturbation_variables_linear_control(
            sensor, ref_inputs, Vel_blc, e_pos_blc, e_chi, X0
        )

        mdes_lon = self.dir_control_lon(state.int_e_long, ctrl_sys.lon, Xlon_cmd, Xlon)
        mdes_lat = self.dir_control_lat(state.int_e_lat, ctrl_sys.lat, Xlat_cmd, Xlat)

        return (X0, U0), ctrl_sys, (mdes_lon, mdes_lat), XLonLat(Xlon, Xlon_cmd, Xlat, Xlat_cmd)

    def dir_control_lon(self, int_e_long: FloatScalar, ctrl_sys: Ctrl_Sys_Lon, R: XLonCmd, X: XLon):
        Ki = ctrl_sys.Ki
        Kx = ctrl_sys.Kx
        G = ctrl_sys.G
        mdes = jnp.dot(G, R).reshape(3) + jnp.dot(Ki, int_e_long) - jnp.dot(Kx, X).reshape(3)
        assert mdes.shape == (3,)
        return mdes

    def dir_control_lat(self, int_e_lat: FloatScalar, ctrl_sys: Ctrl_Sys_Lat, R: XLatCmd, X: XLat):
        """Linear gain depending on ref state and curr state, with an integral term."""
        Ki = ctrl_sys.Ki
        Kx = ctrl_sys.Kx
        G = ctrl_sys.G
        mdes = jnp.dot(G, R).reshape(3) + jnp.dot(Ki, int_e_lat) - jnp.dot(Kx, X).reshape(3)
        assert mdes.shape == (3,)
        return mdes

    def perturbation_variables_linear_control(
        self,
        sensor: SensorNoAcc,
        refInputs: RefInputs,
        Vel_blc: Vec3_1,
        e_pos_blc: Vec3_1,
        e_chi: FloatScalar,
        X0: Vec12_1,
    ) -> [XLon, XLonCmd, XLatCmd, XLat]:
        assert Vel_blc.shape == e_pos_blc.shape == (3, 1)
        Omeg_BIb = sensor.Omeg_BIb

        # [ Roll, Pitch, Yaw ]
        eta = sensor.Euler.as_vec3_1()

        Vel_bIc_cmd = refInputs.Vel_bIc_des.reshape((3, 1))
        chi_dot_des = refInputs.Chi_dot_des
        Vel_bIc_0 = X0[:3]
        Omeg_BIb_0 = X0[3:6]
        # Accel_bIb_0 = X0[6:9]
        eta_0 = X0[9:12]

        pqr_t: Vec3_1 = Omeg_BIb - Omeg_BIb_0
        p_t, q_t, r_t = pqr_t

        Vel_blc = Vel_blc.reshape((3, 1))
        e_pos_blc = e_pos_blc.reshape((3, 1))

        # Note: Represent the CURRENT velocity in control frame after subtracting trim.
        uvw_bar_t: Vec3_1 = Vel_blc - Vel_bIc_0
        ubar_t, vbar_t, wbar_t = uvw_bar_t

        # Note: Desired? velocity in control frame, including a p controller from position.
        Kp_uvw_epos = np.array([0.1, 0.1, 0.1])[:, None]
        uvw_bar_cmd_t = (Vel_bIc_cmd - Vel_bIc_0) + e_pos_blc * Kp_uvw_epos
        ubar_cmd_t, vbar_cmd_t, wbar_cmd_t = uvw_bar_cmd_t

        Kp_chidot_chi = 0.1
        chi_dot_cmd = chi_dot_des + e_chi * Kp_chidot_chi

        # phi (roll) and theta (pitch) after subtracting trim.
        phi_t = (eta - eta_0)[0]
        th_t = (eta - eta_0)[1]

        # import pdb; pdb.set_trace()
        Xlon = jnp.array([ubar_t, wbar_t, q_t, th_t])
        Xlon_cmd = jnp.array([ubar_cmd_t, wbar_cmd_t, [0]])
        Xlat_cmd = jnp.array([vbar_cmd_t, [chi_dot_cmd]])
        Xlat = jnp.array([vbar_t, p_t, r_t, phi_t])

        assert Xlon.shape == (4, 1) and Xlon_cmd.shape == (3, 1)
        assert Xlat.shape == (4, 1) and Xlat_cmd.shape == (2, 1)

        return Xlon, Xlon_cmd, Xlat_cmd, Xlat

    def convert_velocity_position_error_to_control_frame(
        self, sensor: SensorNoAcc, refInputs: RefInputs
    ) -> tuple[Vec3_1, Vec3_1, FloatScalar]:
        Pos_bIi = sensor.Pos_bIi
        Vel_bIi = sensor.Vel_bIi
        eta = sensor.Euler
        chi_des: FloatScalar = refInputs.Chi_des
        # Desired position in world frame?
        pos_des = refInputs.Pos_des.reshape((3, 1))

        # Rotates inertial frame to control frame, i.e., orientation of the inertial frame from the control frame...?
        Q_i2c = quaternion_rotz(chi_des)
        # Q_i2c = jnp.array([[jnp.cos(chi_des / 2)], [0], [0], [jnp.sin(chi_des / 2)]])
        assert Q_i2c.shape == (4, 1)

        # Why is this positive but the below one is negative??
        Vel_blc = quaternion_vector_transformation(Q_i2c, Vel_bIi).reshape((3, 1))

        # Note: Only compute the position error if there is a reference trajectory.
        mul = self.REF_TRAJ_ON or self.POSITION_ERROR

        e_pos_blc = quaternion_vector_transformation(Q_i2c, pos_des - Pos_bIi).reshape((3, 1)) * mul
        # Positive => Psi too small.
        e_chi = chi_des - eta.psi

        return Vel_blc, e_pos_blc, e_chi

    def scheduled_parameters(self, ref_inputs: RefInputs) -> tuple[Vec12_1, Vec12_1, CtrlSys]:
        Vel_blc_cmd = ref_inputs.Vel_bIc_des
        ubar_cmd: FloatScalar = Vel_blc_cmd[0]
        wbar_cmd: FloatScalar = Vel_blc_cmd[2]

        CONDITION = self.REF_TRAJ_ON | self.FEEDBACK_CURRENT
        in_1: FloatScalar = ubar_cmd if CONDITION else 0
        in_2: FloatScalar = wbar_cmd if CONDITION else -8

        # Lin interp in_1.
        k_1, f_1 = get_lininterp_idx(in_1, self.mat["UH"].squeeze(-1))
        k_2, f_2 = get_lininterp_idx(in_2, self.mat["WH"].squeeze(-1))

        # (25, 26, 3)
        table_data = self.mat["XU0_interp"]
        m1 = matrix_interpolation(k_1, f_1, k_2, f_2, table_data)

        m1 = jnp.transpose(m1)
        X0 = m1[:12]
        U0 = m1[12:]
        ctrl_sys_lon = self.longtitudinal_ctrl_interpolation(k_1, f_1, k_2, f_2)
        ctrl_sys_lat = self.lateral_ctrl_interpolation(k_1, f_1, k_2, f_2)
        return X0, U0, CtrlSys(ctrl_sys_lon, ctrl_sys_lat)

    def create_allocation_bus(
        self,
        mdes: Vec6,
        ctrl_sys_lon: Ctrl_Sys_Lon,
        ctrl_sys_lat: Ctrl_Sys_Lat,
        u_lon: Vec11,
        u_lat: Vec10,
        U0: Vec13_1,
        X0: Vec12_1,
    ) -> Alloc:
        B_lon = jnp.zeros((6, 14))
        B_lon = B_lon.at[:3, :12].set(ctrl_sys_lon.B)

        B_lat = jnp.zeros((6, 14))
        B_lat = B_lat.at[:3, :11].set(ctrl_sys_lat.B)

        W_lon = jnp.zeros((14, 14))
        W_lon = W_lon.at[:12, :12].set(ctrl_sys_lon.W)

        W_lat = jnp.zeros((14, 14))
        W_lat = W_lat.at[:11, :11].set(ctrl_sys_lat.W)

        eng_max = np.ones((9, 1)) * 350
        eng_min = np.ones((9, 1)) * (-1e-3)
        act_max = np.ones((5, 1)) * 0.5236
        act_min = np.ones((5, 1)) * (-0.5236)

        u_agi = jnp.zeros((14, 1))
        value_sum = u_lon[:8] + u_lat[:8]
        u_agi = u_agi.at[:8].set(value_sum.reshape((8, 1)))
        u_agi = u_agi.at[8].set(u_lon[8])
        u_agi = u_agi.at[9].set(u_lat[8])
        u_agi = u_agi.at[10].set(-u_lat[8])
        u_agi = u_agi.at[11].set(u_lon[9])
        u_agi = u_agi.at[12].set(u_lon[9])
        u_agi = u_agi.at[13].set(u_lat[9])

        assert mdes.shape == (6,)
        return Alloc(
            mdes=mdes,
            m0_in=np.zeros((6, 1)),
            B_lon=B_lon,
            B_lat=B_lat,
            W_lon=W_lon,
            W_lat=W_lat,
            W_lon_inv=np.zeros((14, 14)),
            W_lat_inv=np.zeros((14, 14)),
            u0_in=np.zeros((14, 1)),
            U_Limits_Upp_in=jnp.concatenate((eng_max, act_max)),
            U_Limits_Lwr_in=jnp.concatenate((eng_min, act_min)),
            u_agi=u_agi,
            m_obt_flag=1,
            exc_lim_flag=0,
            u0=np.zeros((14, 1)),
            u0_trim=U0,
            x0_trim=X0,
        )

    def longtitudinal_ctrl_interpolation(self, k_1, f_1, k_2, f_2):
        Ki = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Ki_lon_interp"])
        Kx = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Kx_lon_interp"])
        Kv = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Kv_lon_interp"])
        F = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["F_lon_interp"])
        G = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["G_lon_interp"])
        C = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["C_lon_interp"])
        Cv = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Cv_lon_interp"])
        W = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["W_lon_interp"])
        B = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["B_lon_interp"])
        Ap = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Ap_lon_interp"])
        Bp = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Bp_lon_interp"])
        return Ctrl_Sys_Lon(Ki=Ki, Kx=Kx, Kv=Kv, F=F, G=G, C=C, Cv=Cv, W=W, B=B, Ap=Ap, Bp=Bp)

    def lateral_ctrl_interpolation(self, k_1, f_1, k_2, f_2):
        Ki = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Ki_lat_interp"])
        Kx = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Kx_lat_interp"])
        Kv = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Kv_lat_interp"])
        F = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["F_lat_interp"])
        G = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["G_lat_interp"])
        C = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["C_lat_interp"])
        Cv = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Cv_lat_interp"])
        W = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["W_lat_interp"])
        B = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["B_lat_interp"])
        Ap = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Ap_lat_interp"])
        Bp = matrix_interpolation(k_1, f_1, k_2, f_2, self.mat["Bp_lat_interp"])
        return Ctrl_Sys_Lat(Ki=Ki, Kx=Kx, Kv=Kv, F=F, G=G, C=C, Cv=Cv, W=W, B=B, Ap=Ap, Bp=Bp)
