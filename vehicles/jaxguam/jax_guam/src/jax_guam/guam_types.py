from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from jax_guam.utils.jax_types import (
    FloatScalar,
    Mat1_4,
    Mat3,
    Mat3_4,
    Mat3_11,
    Mat3_12,
    Mat4,
    Mat4_11,
    Mat6_14,
    Mat11,
    Mat12,
    Mat14,
    Mat31,
    Quat,
    Quat_1,
    Vec3,
    Vec3_1,
    Vec5,
    Vec5_1,
    Vec6,
    Vec9,
    Vec9_1,
    Vec12,
    Vec13,
    Vec13_1,
    Vec14,
    Vec15,
)


######### power system #########
class PwrCmd(NamedTuple):
    CtrlSurfacePwr: Vec5
    EnginePwr: Vec9


class Power(NamedTuple):
    CtrlSurfacePwr: Vec5
    EnginePwr: Vec9


EngineControl = Vec9_1
EngineControl_1 = Vec9_1
SurfControl = Vec5
SurfControl_1 = Vec5_1

######### surf eng #########
class Cmd(NamedTuple):  # Note 9x1, 9, 5x1, 5
    EngineCmd: EngineControl
    EnginePwr: Vec9_1
    CtrlSurfaceCmd: Vec5_1
    CtrlSurfacePwr: Vec5_1
    GearCmd: FloatScalar


class Failure_Surfaces(NamedTuple):
    F_Fail_Initiate: Vec5
    F_Hold_Last: Vec5
    F_Pre_Scale: Vec5
    F_Post_Scale: Vec5
    F_Pos_Bias: Vec5
    F_Pos_Scale: Vec5
    F_Up_Plim: Vec5
    F_Lwr_Plim: Vec5
    F_Rate_Bias: Vec5
    F_Rate_Scale: Vec5
    F_Up_Rlim: Vec5
    F_Lwr_Rlim: Vec5
    F_Accel_Bias: Vec5
    F_Accel_Scale: Vec5
    F_Up_Alim: Vec5
    F_Lwr_Alim: Vec5
    F_Gen_Sig_Sel: Vec15


class TraditionalSurfaces(NamedTuple):
    aileron: FloatScalar
    flap: FloatScalar
    elevator: FloatScalar
    rudder: FloatScalar


class SurfAct(NamedTuple):
    # [flaperon L, flaperon R, elevator L, elevator R, rudder]
    CtrlSurfPos: Vec5
    # Note: Not actually used.
    CtrlSurfRate: Vec5_1
    # Note: Not actually used.
    Failure_Surfaces: Failure_Surfaces


class Failure_Engines(NamedTuple):
    F_Fail_Initiate: Vec9
    F_Hold_Last: Vec9
    F_Pre_Scale: Vec9
    F_Post_Scale: Vec9
    F_Pos_Bias: Vec9
    F_Pos_Scale: Vec9
    F_Up_Plim: Vec9
    F_Lwr_Plim: Vec9
    F_Rate_Bias: Vec9
    F_Rate_Scale: Vec9
    F_Up_Rlim: Vec9
    F_Lwr_Rlim: Vec9
    F_Accel_Bias: Vec9
    F_Accel_Scale: Vec9
    F_Up_Alim: Vec9
    F_Lwr_Alim: Vec9
    F_Gen_Sig_Sel: Vec15


class PropAct(NamedTuple):
    EngSpeed: Vec9
    EngAccel: Vec9
    Failure_Engines: Failure_Engines


class TotalFM(NamedTuple):
    Forces_b: Vec3_1
    Moments_b: Vec3_1
    H_b: Vec3_1
    Hdot_b: Vec3_1


class AeroFM(NamedTuple):
    Aero_Fb: Vec3_1
    Aero_Mb: Vec3_1


class PropFM(NamedTuple):
    Prop_Fb: Vec3_1
    Prop_Mb: Vec3_1
    Prop_T: Vec9
    Prop_Q: Vec9


class FM(NamedTuple):
    TotalFM: TotalFM
    # Wing, Tail and Body
    AeroFM: AeroFM
    # Propellers
    PropFM: PropFM


class GUAMState(NamedTuple):
    pos_bii: Vec3
    vel_bIi: Vec3
    # Depends on derivs. Used by sensors.
    accel_bii: Vec3
    # Note: Depends on derivs. Not used elsewhere.
    asensed_bIb: Vec3
    Q_i2b: Quat
    Omega_BIb: Vec3
    # Depends on derivs.
    OmegaDtl_BIb: Vec3


AircraftStateVec = Vec13
AircraftStateVec_1 = Vec13_1


class AircraftState(NamedTuple):
    vel: Vec3_1
    omega: Vec3_1
    pos: Vec3_1
    quat: Quat_1

    @property
    def VelDtB_bEb(self):
        """Velocity dt body?"""
        return self.vel

    @property
    def OmgDtI_BIb(self):
        """Omega dt inertial?"""
        return self.omega

    @property
    def PosDtI_bii(self):
        """Position dt inertial?"""
        return self.pos

    @property
    def QDt_i2b(self):
        """Quat dt inertial to body?"""
        return self.quat

    @staticmethod
    def GetDefault13() -> AircraftStateVec:
        x = np.array([-0.00069017, 0, -8, 0, 0, 0, 0, 0, 0, 1, 0, -4.3136e-05, 0])
        assert x.shape == (13,)
        return x

    @staticmethod
    def from_state13(state13: AircraftStateVec | AircraftStateVec_1):
        if state13.shape == (13,):
            state13 = state13[:, None]
        assert state13.shape == (13, 1)
        return AircraftState(state13[:3], state13[3:6], state13[6:9], state13[9:13]).verify()

    @staticmethod
    def from_state13_batch(state13: AircraftStateVec):
        assert state13.shape[-1] == 13
        return AircraftState(state13[..., :3], state13[..., 3:6], state13[..., 6:9], state13[..., 9:13])

    @staticmethod
    def createDeriv(VelDtB_bEb: Vec3_1, OmgDtI_BIb: Vec3_1, PosDtI_bii: Vec3_1, QDt_i2b: Quat_1):
        return AircraftState(VelDtB_bEb, OmgDtI_BIb, PosDtI_bii, QDt_i2b).verify()

    def verify(self) -> "AircraftState":
        assert self.VelDtB_bEb.shape == (3, 1)
        assert self.OmgDtI_BIb.shape == (3, 1)
        assert self.PosDtI_bii.shape == (3, 1)
        assert self.QDt_i2b.shape == (4, 1)
        return self


class Wind(NamedTuple):
    Vel_wHh: Vec3
    Vel_DtH_wHh: Vec3


class Turbulence(NamedTuple):
    Vel_tBb: Vec3
    VelDtB_tBb: Vec3
    Omeg_TBb: Vec3
    OmegDtB_TBb: Vec3


class Atmosphere(NamedTuple):
    Density: FloatScalar
    Pressure: FloatScalar
    Temperature: FloatScalar
    SpeedOfSound: FloatScalar


class EnvData(NamedTuple):
    Wind: Wind
    Turbulence: Turbulence
    Atmosphere: Atmosphere


class EulerAngles(NamedTuple):
    # Roll
    phi: FloatScalar
    # Pitch
    theta: FloatScalar
    # Yaw
    psi: FloatScalar

    def as_vec3_1(self) -> Vec3_1:
        assert self.phi.shape == self.theta.shape == self.psi.shape == tuple()
        return jnp.array([[self.phi], [self.theta], [self.psi]])


class LatLonAlt(NamedTuple):
    LatGeod: FloatScalar
    Lon: FloatScalar
    AltGeod: FloatScalar


class WorldRel(NamedTuple):
    """Only used in Env and AeroProp. Rest not used at all."""

    Pos_bee: Vec3 = None
    Pos_beh_topo: Vec3 = None
    # Note: Used in Sensors, but unused afterwards.
    LatLonAlt: FloatScalar = None
    # Note: Used in Env.
    AltMSL: FloatScalar = None
    AltAGL: FloatScalar = None
    AltPresMSL: FloatScalar = None
    Euler: FloatScalar = None
    # Note :Used in Env.
    Q_h2b: Quat = None
    Q_i2h: Quat = None
    gamma: FloatScalar = None
    chi: FloatScalar = None
    # Note: Used in air_rel
    Omeg_BEb: Vec3 = None
    # Note: Used in air_rel
    Omeg_BHb: Vec3 = None
    # Note: Used in Env.
    Vel_bEb: Vec3 = None
    Vel_bEh: Vec3 = None
    Omeg_HEh: Vec3 = None
    # Note: Used in air_rel
    VelDtE_bEb: Vec3 = None
    VelDtH_bEh: Vec3 = None
    gammadot: FloatScalar = None
    chidot: FloatScalar = None

    @staticmethod
    def CreateUseful(AltMSL: FloatScalar, Q_h2b: Quat_1, Vel_bEb: Vec3):
        return WorldRel(AltMSL=AltMSL, Q_h2b=Q_h2b, Vel_bEb=Vel_bEb)


class AirRel(NamedTuple):
    Vel_bWb: Vec3 = None
    VelDtB_bWb: Vec3 = None
    Omeg_BWb: Vec3 = None
    # Note: Used in sensors.
    Vtot: FloatScalar = None
    Veas: FloatScalar = None
    Vcas: FloatScalar = None
    Mach: FloatScalar = None
    qbar: FloatScalar = None
    qc: FloatScalar = None
    alpha: FloatScalar = None
    beta: FloatScalar = None
    mu: FloatScalar = None
    # Note: Used in sensors.
    gamma: FloatScalar = None
    # Note: Used in sensors.
    chi: FloatScalar = None
    Vtotdot: FloatScalar = None
    alphadot: FloatScalar = None
    betadot: FloatScalar = None
    mudot: FloatScalar = None
    gammadot: FloatScalar = None
    chidot: FloatScalar = None
    alphaTotal: FloatScalar = None
    phiAero: FloatScalar = None
    Q_h2v: Quat = None
    Omeg_VHb: Vec3 = None

    @staticmethod
    def CreateUseful(Vtot: FloatScalar, gamma: FloatScalar, chi: FloatScalar):
        return AirRel(Vtot=Vtot, gamma=gamma, chi=chi)


class AeroPropBodyData(NamedTuple):
    vel_bIi: Vec3_1
    Omega_BIb: Vec3_1
    Q_i2b: Vec3_1
    Q_h2b: Vec3_1


class EOM(NamedTuple):
    InertialData: GUAMState
    # Only used in Env and AeroProp (Q_h2b)
    WorldRelativeData: WorldRel
    # Only used in sensors for Vtot, gamma, chi.
    AirRelativeData: AirRel

    def get_aeroprop_body_data(self) -> AeroPropBodyData:
        return AeroPropBodyData(
            self.InertialData.vel_bIi,
            self.InertialData.Omega_BIb,
            self.InertialData.Q_i2b,
            self.WorldRelativeData.Q_h2b,
        )


class SensorNoAcc(NamedTuple):
    # First, the state of GUAM, in the order of AircraftState
    Vel_bIi: Vec3_1
    Omeg_BIb: Vec3
    Pos_bIi: Vec3_1
    Q_i2b: Quat_1
    # For convenience, euler angle of the above.
    Euler: EulerAngles
    # Note: things below require acceleration!
    # Vtot: FloatScalar
    # gamma: FloatScalar
    # chi: FloatScalar


class Sensor(NamedTuple):
    # Note: Computed from GUAMState.Omega_BIb, USED IN LCControl in perturb_variables_linear_control.
    Omeg_BIb: Vec3
    # Note: Computed from GUAMState.accel_bii, not used elsewhere. Accel requires forces, which requires control.
    Accel_bIb: Vec3
    # Orientation, i.e., inertial to body?
    Q_i2b: Quat
    # Position in world frame?
    Pos_bIi: Vec3_1
    # Velocity in world frame?
    Vel_bIi: Vec3_1
    # Note: Taken from WorldRelativeData. Not used elsewhere.
    gpsLLA: Vec3_1
    # Not used elsewhere.
    LaserAlt: FloatScalar
    Euler: EulerAngles
    # Note: Taken from AirRelativeData, not used elsewhere.
    Vtot: FloatScalar
    # Note: Taken from AirRelativeData, not used elsewhere.
    gamma: FloatScalar
    # Note: Taken from AirRelativeData, not used elsewhere.
    chi: FloatScalar


class VehicleOut(NamedTuple):
    Power: Power
    SurfAct: SurfAct
    PropAct: PropAct
    Gear: FloatScalar
    FM: FM
    EOM: EOM
    Sensor: Sensor


class RefInputs(NamedTuple):
    Vel_bIc_des: Vec3
    Pos_des: Vec3
    # χ from the wind rotation-anges? i.e., [μ γ χ], χ is the "heading" of the velocity.
    Chi_des: FloatScalar
    Chi_dot_des: FloatScalar

    def assert_shapes(self):
        assert self.Vel_bIc_des.shape == (3,)
        assert self.Pos_des.shape == (3,)
        assert isinstance(self.Chi_des, (int, float)) or self.Chi_des.shape == tuple()
        assert isinstance(self.Chi_dot_des, (int, float)) or self.Chi_dot_des.shape == tuple()


class TrimInputs(NamedTuple):
    Engines: Vec9
    Surfaces: Quat


class Ctrl_Sys_Lon(NamedTuple):
    # Gain for the error integral, Used for the output.
    Ki: Mat3
    # Gain for the current longitudinal state, Used for the output.
    Kx: Mat3_4
    # Gain for the feedback θ pitch, ONLY USED FOR INTEGRAL COMPUTATION.
    Kv: Mat31
    # Gain for the ref longitudinal state, ONLY USED FOR INTEGRAL COMPUTATION.
    F: Mat3
    # Gain for the ref longitudinal state, Used for the output.
    G: Mat3
    # "Negative" Gain for the current longitudinal state, ONLY USED FOR INTEGRAL COMPUTATION.
    C: Mat3_4
    # Transforms longitudinal state to units of "feedback" = θ pitch, ONLY USED FOR INTEGRAL COMPUTATION.
    Cv: Mat1_4
    # Weighting matrix for each of the 11 + 1 longitudinal controls. [ ω_r (8,); ω_p; elevator; flaps; theta ]
    W: Mat12
    # Maps from each of the 11 + 1 longitudinal controls to the 3 longitudinal controls? maybe [ F_x, F_z, τ_y ]?
    B: Mat3_12
    # Note: following two matrices are not used.
    Ap: Mat4
    Bp: Mat4_11


class Ctrl_Sys_Lat(NamedTuple):
    Ki: Mat3
    Kx: Mat3_4
    Kv: Mat31
    F: Mat3
    G: Mat3
    C: Mat3_4
    Cv: Mat1_4
    # Weighting matrix for each of the 10 + 1 lateral controls. [ ω_r (8,); aileron; rudder; phi ]
    W: Mat11
    # Maps from each of the 10 + 1 lateral controls to the 3 lateral controls? maybe [ F_y, τ_x, τ_z ]?
    B: Mat3_11
    # Note: following two matrices are not used.
    Ap: Mat4
    Bp: Mat4_11


class CtrlSys(NamedTuple):
    lon: Ctrl_Sys_Lon
    lat: Ctrl_Sys_Lat


class Alloc(NamedTuple):
    mdes: Vec6
    m0_in: Vec6
    B_lon: Mat6_14
    B_lat: Mat6_14
    W_lon: Mat14
    W_lat: Mat14
    W_lon_inv: Mat14
    W_lat_inv: Mat14
    u0_in: Vec14
    U_Limits_Upp_in: Vec14
    U_Limits_Lwr_in: Vec14
    u_agi: Vec14
    m_obt_flag: FloatScalar
    exc_lim_flag: FloatScalar
    u0: Vec14
    u0_trim: Vec13
    x0_trim: Vec12


class Control(NamedTuple):
    Cmd: Cmd
    Alloc: Alloc


class SimOutputs(NamedTuple):
    Env: EnvData
    Vehicle: VehicleOut
    Control: Control
    RefInputs: RefInputs
    Time: FloatScalar
    # InvalidData:


class Trim(NamedTuple):
    ...
