from typing import NamedTuple
from loguru import logger

import ipdb
import jax.numpy as jnp
import numpy as np
from extensisq import BS5
from scipy.integrate import solve_ivp

from jax_guam.data.read_data import read_data
from jax_guam.guam_types import (
    Alloc,
    Cmd,
    Control,
    CtrlSys,
    Ctrl_Sys_Lat,
    Ctrl_Sys_Lon,
    EngineControl,
    EngineControl_1,
    RefInputs,
    Sensor,
)
from jax_guam.utils.functions import (
    matrix_interpolation,
    pseudo_inverse,
    quaternion_rotz,
    quaternion_vector_transformation, get_lininterp_idx,
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
    Vec13,
    Vec13_1,
    Vec25_1,
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


class L_C_control:
    def __init__(self):
        self.mat = read_data(data_dir() / "trim_table.mat")
        # self.feedback = [None, None]  Note: Not needed as a state.
        self.long_inte = jnp.array([0, 0, 0])
        self.lat_inte = jnp.array([0, 0, 0])

        self.MAX_POS_ENGINE = np.full((9, 1), 350.0)
        self.MIN_POS_ENGINE = np.full((9, 1), -1e-3)

    def lift_cruise_control(self, time, sensor: Sensor, refInputs: RefInputs) -> Control:
        # Note: Time is not used for anything. Controls are time-invariant.

        # Note: Uses self.long_inte and self.lat_inte inside.
        # Note: Also, self.feedback not actually used here, so it was removed.
        Xu_IC, ctrl_sys, X, mdes, e_pos_blc, X_all = self.baseline(sensor, refInputs, None)

        adaptive = jnp.zeros(6)
        m_adapt = mdes + adaptive

        engCmd, surfCmd, feedback, u_alloc = self.pseudo_inverse_control_allocation(Xu_IC, ctrl_sys, m_adapt)
        # logger.info("mdes: {}, eng: {}, surf: {}".format(mdes.squeeze(), engCmd.squeeze(), surfCmd.squeeze()))
        # ipdb.set_trace()
        cmd = Cmd(
            EngineCmd=engCmd,
            EnginePwr=np.ones((9, 1)),
            CtrlSurfaceCmd=surfCmd,
            CtrlSurfacePwr=np.ones((5, 1)),
            GearCmd=1,
        )
        control = Control(Cmd=cmd, Alloc=u_alloc)

        ###################################################################################
        # Everything below this is dynamics update.
        Xlon, Xlon_cmd, Xlat, Xlat_cmd = X_all

        # Note: Modifies self.long_dot
        self.directional_control_long(feedback[0], ctrl_sys.lon, Xlon, Xlon_cmd)
        # Note: Modifies self.lat_dot
        self.directional_control_lat(feedback[1], ctrl_sys.lat, Xlat, Xlat_cmd)

        # Integration occurs over slef.long_dot and self.lat_dot.
        sol_long_init = self.long_inte
        sol_long = solve_ivp(
            self.directional_control_long_inte,
            [time, time + 0.005],
            sol_long_init,
            t_eval=[time + 0.005],
            method=BS5,
        )

        sol_lat_init = self.lat_inte
        sol_lat = solve_ivp(
            self.directional_control_lat_inte,
            [time, time + 0.005],
            sol_lat_init,
            t_eval=[time + 0.005],
            method=BS5,
        )
        self.long_inte = sol_long.y.reshape(3)
        self.lat_inte = sol_lat.y.reshape(3)
        ###################################################################################

        return control

    def baseline(self, sensor: Sensor, refInputs: RefInputs, feedback: Vec2):
        # theta_fb = feedback[0]
        # phi_fb = feedback[1]

        Vel_blc, e_pos_blc, e_chi = self.convert_velocity_position_error_to_control_frame(sensor, refInputs)

        # Note: We change to explicitly passing ctrl_sys_lon and ctrl_sys_lat.
        X0, U0, ctrl_sys_lon, ctrl_sys_lat = self.scheduled_parameters(refInputs)

        # X0: (12, 1) = [ vel_blc (3) , ω_blc (3), a_blc (3), rpy (3) ].

        # Note: Previously, self.X0 and self.U0 were set, but these are not used outside.
        # self.X0, self.U0 = X0, U0

        Xlon, Xlon_cmd, Xlat_cmd, Xlat = self.perturbation_variables_linear_control(
            sensor, refInputs, Vel_blc, e_pos_blc, e_chi, X0
        )

        # Note: feedback not actually used in the init?
        theta_fb, phi_fb = None, None
        mdes_lon = self.directional_control_long_init(ctrl_sys_lon, Xlon_cmd, Xlon, theta_fb)
        mdes_lat = self.directional_control_lat_init(ctrl_sys_lat, Xlat_cmd, Xlat, phi_fb)

        # logger.info("mdes lon: {}, lat: {}".format(mdes_lon.squeeze(), mdes_lat.squeeze()))
        # ipdb.set_trace()

        # import pdb; pdb.set_trace()
        XU_IC = jnp.concatenate([X0, U0])
        ctrl_sys = CtrlSys(lon=ctrl_sys_lon, lat=ctrl_sys_lat)
        X = jnp.concatenate([Xlon, Xlat])
        mdes = jnp.concatenate([mdes_lon, mdes_lat])

        X_all = (Xlon, Xlon_cmd, Xlat, Xlat_cmd)

        return XU_IC, ctrl_sys, X, mdes, e_pos_blc, X_all

    def convert_velocity_position_error_to_control_frame(
        self, sensor: Sensor, refInputs: RefInputs
    ) -> tuple[Vec3_1, Vec3_1, FloatScalar]:
        RefTrajOn = 1
        PositionError = 0
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
        mul = RefTrajOn if RefTrajOn > 0 else PositionError

        e_pos_blc = quaternion_vector_transformation(Q_i2c, pos_des - Pos_bIi).reshape((3, 1)) * mul
        # Positive => Psi too small.
        e_chi = chi_des - eta.psi

        return Vel_blc, e_pos_blc, e_chi

    def scheduled_parameters(self, refInputs: RefInputs) -> tuple[Vec12_1, Vec12_1, Ctrl_Sys_Lon, Ctrl_Sys_Lat]:
        Vel_blc_cmd = refInputs.Vel_bIc_des
        ubar_cmd: FloatScalar = Vel_blc_cmd[0]
        wbar_cmd: FloatScalar = Vel_blc_cmd[2]
        RefTrajOn = 1
        FeedbackCurrent = 1
        condition = RefTrajOn if RefTrajOn > 0 else FeedbackCurrent
        in_1: FloatScalar = ubar_cmd if condition != 0 else 0
        in_2: FloatScalar = wbar_cmd if condition != 0 else -8

        # (26,)
        values_1 = jnp.array(self.mat["UH"])
        values_1 = values_1.reshape(values_1.shape[0])
        assert values_1.shape == (26,)

        # (3,)
        values_2 = jnp.array(self.mat["WH"])
        values_2 = values_2.reshape(values_2.shape[0])
        assert values_2.shape == (3,)

        # Note: Linear interp. k_1 and k_2 are the idxs, f_1 and f_2 are the interpolation factors.
        k_1, f_1 = get_lininterp_idx(in_1, values_1)
        k_2, f_2 = get_lininterp_idx(in_2, values_2)
        # if in_1 < values_1[0]:
        #     k_1 = 0
        #     f_1 = 0
        # elif in_1 > values_1[-1]:
        #     k_1 = values_1.shape[0] - 1
        #     f_1 = 0
        # else:
        #     k_1 = jnp.digitize(in_1, values_1, right=False) - 1
        #     f_1 = (in_1 - values_1[k_1]) / (values_1[k_1 + 1] - values_1[k_1])
        #
        # if in_2 < values_2[0]:
        #     k_2 = 0
        #     f_2 = 0
        # elif in_2 > values_2[-1]:
        #     k_2 = values_2.shape[0] - 1
        #     f_2 = 0
        # else:
        #     k_2 = jnp.digitize(in_2, values_2, right=False) - 1
        #     f_2 = (in_2 - values_2[k_2]) / (values_2[k_2 + 1] - values_2[k_2])

        # (25, 26, 3)
        table_data = self.mat["XU0_interp"]
        m1 = matrix_interpolation(k_1, f_1, k_2, f_2, table_data)

        m1 = jnp.transpose(m1)
        X0 = m1[:12]
        U0 = m1[12:]
        ctrl_sys_lon = self.longtitudinal_ctrl_interpolation(k_1, f_1, k_2, f_2)
        ctrl_sys_lat = self.lateral_ctrl_interpolation(k_1, f_1, k_2, f_2)
        # pdb.set_trace()
        return X0, U0, ctrl_sys_lon, ctrl_sys_lat

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

    def perturbation_variables_linear_control(
        self, sensor: Sensor, refInputs: RefInputs, Vel_blc: Vec3_1, e_pos_blc: Vec3_1, e_chi: FloatScalar, X0: Vec12_1
    ) -> [XLon, XLonCmd, XLatCmd, XLat]:
        assert Vel_blc.shape == e_pos_blc.shape == (3, 1)
        Omeg_BIb = sensor.Omeg_BIb

        # [ Roll, Pitch, Yaw ]
        # eta = sensor.Euler
        # eta = jnp.array([[eta.phi], [eta.theta], [eta.psi]])
        eta = sensor.Euler.as_vec3_1()

        Vel_bIc_cmd = refInputs.Vel_bIc_des.reshape((3, 1))
        chi_dot_des = refInputs.Chi_dot_des
        Vel_bIc_0 = X0[:3]
        Omeg_BIb_0 = X0[3:6]
        # Accel_bIb_0 = X0[6:9]
        eta_0 = X0[9:12]

        # p_t = (Omeg_BIb - Omeg_BIb_0)[0]
        # q_t = (Omeg_BIb - Omeg_BIb_0)[1]
        # r_t = (Omeg_BIb - Omeg_BIb_0)[2]
        pqr_t: Vec3_1 = Omeg_BIb - Omeg_BIb_0
        p_t, q_t, r_t = pqr_t

        Vel_blc = Vel_blc.reshape((3, 1))
        e_pos_blc = e_pos_blc.reshape((3, 1))

        # Note: Represent the CURRENT velocity in control frame after subtracting trim.
        # ubar_t = (Vel_blc - Vel_bIc_0)[0]
        # vbar_t = (Vel_blc - Vel_bIc_0)[1]
        # wbar_t = (Vel_blc - Vel_bIc_0)[2]
        uvw_bar_t: Vec3_1 = Vel_blc - Vel_bIc_0
        ubar_t, vbar_t, wbar_t = uvw_bar_t

        # Note: Desired? velocity in control frame, including a p controller from position.
        # ubar_cmd_t = (Vel_bIc_cmd - Vel_bIc_0 + e_pos_blc * jnp.array([[0.1], [0.1], [0.1]]))[0]
        # vbar_cmd_t = (Vel_bIc_cmd - Vel_bIc_0 + e_pos_blc * jnp.array([[0.1], [0.1], [0.1]]))[1]
        # wbar_cmd_t = (Vel_bIc_cmd - Vel_bIc_0 + e_pos_blc * jnp.array([[0.1], [0.1], [0.1]]))[2]
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

    def directional_control_long_init(self, ctrl_sys: Ctrl_Sys_Lon, R: XLonCmd, X: XLon, feedback):
        """ Linear gain depending on ref state and curr state, with an integral term.
        """
        Ki = ctrl_sys.Ki
        Kx = ctrl_sys.Kx
        G = ctrl_sys.G
        mdes = jnp.dot(G, R).reshape(3) + jnp.dot(Ki, self.long_inte) - jnp.dot(Kx, X).reshape(3)
        assert mdes.shape == (3,)
        return mdes

    def directional_control_lat_init(self, ctrl_sys: Ctrl_Sys_Lat, R: XLatCmd, X: XLat, feedback):
        """ Linear gain depending on ref state and curr state, with an integral term.
        """
        Ki = ctrl_sys.Ki
        Kx = ctrl_sys.Kx
        G = ctrl_sys.G
        mdes = jnp.dot(G, R).reshape(3) + jnp.dot(Ki, self.lat_inte) - jnp.dot(Kx, X).reshape(3)
        assert mdes.shape == (3,)
        return mdes

    def directional_control_long(
        self, feedback: FloatScalar, ctrl_sys_lon: Ctrl_Sys_Lon, Xlon: Vec4_1, Xlon_cmd: Vec3_1
    ):
        """Error used for the integral term."""
        # Note: Previously, self.ctrl_sys_lon was used.
        Kv = ctrl_sys_lon.Kv
        F = ctrl_sys_lon.F
        C = ctrl_sys_lon.C
        Cv = ctrl_sys_lon.Cv
        self.long_dot = jnp.dot(F, Xlon_cmd) - jnp.dot(C, Xlon) + jnp.dot(Kv, (feedback - jnp.dot(Cv, Xlon)))
        return self.long_dot

    # def directional_control_long_inte(self, time, y):
    #     Ki = self.ctrl_sys_lon.Ki
    #     Kx = self.ctrl_sys_lon.Kx
    #     G = self.ctrl_sys_lon.G
    #     mdes_lon = jnp.dot(G, self.Xlon_cmd).reshape(3) + jnp.dot(Ki, y) - jnp.dot(Kx, self.Xlon).reshape(3)
    #     M_lon = pseudo_inverse(self.ctrl_sys_lon.W, self.ctrl_sys_lon.B)
    #     out_lon = jnp.dot(M_lon, mdes_lon)
    #     theta = out_lon[-1]
    #     Kv = self.ctrl_sys_lon.Kv
    #     F = self.ctrl_sys_lon.F
    #     C = self.ctrl_sys_lon.C
    #     Cv = self.ctrl_sys_lon.Cv
    #     pdb.set_trace()
    #     long_dot = (jnp.dot(F, self.Xlon_cmd) - jnp.dot(C, self.Xlon) + jnp.dot(Kv, (theta - jnp.dot(Cv, self.Xlon)))).reshape(3)
    #     return long_dot

    def directional_control_long_inte(self, time, y):
        return self.long_dot.reshape(3)

    def directional_control_lat(
        self, feedback: FloatScalar, ctrl_sys_lat: Ctrl_Sys_Lat, Xlat: Vec4_1, Xlat_cmd: Vec2_1
    ):
        Kv = ctrl_sys_lat.Kv
        F = ctrl_sys_lat.F
        C = ctrl_sys_lat.C
        Cv = ctrl_sys_lat.Cv
        self.lat_dot = jnp.dot(F, Xlat_cmd) - jnp.dot(C, Xlat) + jnp.dot(Kv, (feedback - jnp.dot(Cv, Xlat)))
        return self.lat_dot

    # def directional_control_lat_inte(self, time, y):
    #     Ki = self.ctrl_sys_lat.Ki
    #     Kx = self.ctrl_sys_lat.Kx
    #     G = self.ctrl_sys_lat.G
    #     mdes_lat = jnp.dot(G, self.Xlat_cmd).reshape(3) + jnp.dot(Ki, y) - jnp.dot(Kx, self.Xlat).reshape(3)
    #     XU_IC = jnp.concatenate([self.X0, self.U0])
    #     ctrl_sys = Ctrl_Sys(ctrl_sys_lon = self.ctrl_sys_lon, ctrl_sys_lat = self.ctrl_sys_lat)
    #     X = jnp.concatenate([self.Xlon, self.Xlat])
    #     mdes = jnp.concatenate([self.mdes_lon, mdes_lat])
    #     adaptive = jnp.zeros(6)
    #     m_adapt = mdes + adaptive
    #     engCmd, surfCmd, feedback, u_alloc = self.pseudo_inverse_control_allocation(XU_IC, ctrl_sys, m_adapt)
    #     return self.directional_control_lat(feedback[1]).reshape(3)

    def directional_control_lat_inte(self, time, y):
        return self.lat_dot.reshape(3)

    def pseudo_inverse_control_allocation(self, XU_IC: Vec25_1, ctrl_sys: CtrlSys, mdes: Vec6):
        # import pdb; pdb.set_trace()
        X0: Vec12_1 = XU_IC[:12]
        U0: Vec13_1 = XU_IC[12:]
        ctrl_sys_lon = ctrl_sys.lon
        ctrl_sys_lat = ctrl_sys.lat
        mdes_lon: Vec3 = mdes[:3]
        mdes_lat: Vec3 = mdes[3:]
        M_lon = pseudo_inverse(ctrl_sys_lon.W, ctrl_sys_lon.B)
        M_lat = pseudo_inverse(ctrl_sys_lat.W, ctrl_sys_lat.B)
        assert M_lon.shape == (12, 3)
        assert M_lat.shape == (11, 3)

        # (12, )
        out_lon = jnp.dot(M_lon, mdes_lon)
        # (11, )
        out_lat = jnp.dot(M_lat, mdes_lat)
        assert out_lon.shape == (12,) and out_lat.shape == (11,)

        # lon: (11, ) = [ ω_r (8,); ω_p; elevator; flaps ]
        # lat: (12, ) = [ ω_r (8,); aileron; rudder ]

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

        u_alloc = self.create_allocation_bus(mdes, ctrl_sys_lon, ctrl_sys_lat, u_lon, u_lat, U0, X0)
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

        # engBias = U0[4:]
        # a_max = jnp.ones((9, 1)) * 350
        # a_min = jnp.ones((9, 1)) * (-1e-3)
        # out = jnp.clip(jnp.concatenate([jnp.array(om_r), jnp.array([om_p])]).reshape((9, 1)) + engBias, a_min, a_max)
        # print("out.shape: {}".format(out.shape))
        # return out

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
        bias3: Vec1_1 = np.array([U0[3]])
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
