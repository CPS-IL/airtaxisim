import ipdb
import numpy as np

from jax_guam.guam_types import *
from jax_guam.utils.functions import (
    quaternion_conjugate,
    quaternion_multiplication,
    quaternion_to_DCM,
    quaternion_vector_transformation,
    single_axis_quaternion,
)
from jax_guam.utils.jax_types import Vec1


def get_Q2hb_AltMSL(vel_bIi, Omega_BIb, pos_bii, Q_i2b) -> tuple[Quat_1, FloatScalar]:
    Omeg_EIi = np.array([[0], [0], [0]])

    Pos_bei, Vel_bEi, Omeg_BEb = _ECEF_noacc(pos_bii, vel_bIi, Omega_BIb, Q_i2b, Omeg_EIi)
    Vel_bEh, Omeg_BHb, Q_h2b, AltGeod = _get_local_ned_flattened_useful(Pos_bei, Vel_bEi, Omeg_BEb, Q_i2b)
    AltMSL = AltGeod

    return Q_h2b, AltMSL


def get_world_air_rel_simple(inertial_data: GUAMState, env: EnvData) -> tuple[WorldRel, AirRel]:
    pos_bii = inertial_data.pos_bii
    vel_bIi = inertial_data.vel_bIi
    accel_bii = inertial_data.accel_bii
    Omega_BIb = inertial_data.Omega_BIb
    Q_i2b = inertial_data.Q_i2b

    Omeg_EIi = np.array([[0], [0], [0]])
    OmegDtI_EIi = np.array([[0], [0], [0]])

    Pos_bei, Vel_bEi, VelDtE_bEi, Omeg_BEb = _ECEF(pos_bii, vel_bIi, accel_bii, Omega_BIb, Q_i2b, Omeg_EIi, OmegDtI_EIi)

    Vel_bEb = quaternion_vector_transformation(Q_i2b, Vel_bEi).reshape((3, 1))

    Vel_bEh, Omeg_BHb, Q_h2b, AltGeod = _get_local_ned_flattened_useful(Pos_bei, Vel_bEi, Omeg_BEb, Q_i2b)

    AltMSL = AltGeod

    VelDtE_bEb = quaternion_vector_transformation(Q_i2b, VelDtE_bEi).reshape((3, 1))

    #############################################################################################3

    Vel_tBb = env.Turbulence.Vel_tBb
    VelDtB_tBb = env.Turbulence.VelDtB_tBb
    Omeg_TBb = env.Turbulence.Omeg_TBb
    Vel_wHh = env.Wind.Vel_wHh
    Vel_DtH_wHh = env.Wind.Vel_DtH_wHh

    Vel_wEb = quaternion_vector_transformation(Q_h2b, Vel_wHh).reshape((3, 1))
    VelDtH_wEb = quaternion_vector_transformation(Q_h2b, Vel_DtH_wHh).reshape((3, 1))

    Vel_bWb, VelDtB_bWb, Omeg_BWb = _air_relative(
        Vel_tBb, VelDtB_tBb, Omeg_TBb, Vel_wEb, VelDtH_wEb, Vel_bEb, VelDtE_bEb, Omeg_BEb, Omeg_BHb
    )
    Vtot = jnp.sqrt(jnp.sum(Vel_bWb**2))
    chi, gamma = _flight_path_angles(Vel_bEh)
    return WorldRel.CreateUseful(AltMSL, Q_h2b, Vel_bEb), AirRel.CreateUseful(Vtot, gamma, chi)


def get_world_rel(inertial_data: GUAMState, env: EnvData) -> WorldRel:
    pos_bii = inertial_data.pos_bii
    vel_bIi = inertial_data.vel_bIi
    accel_bii = inertial_data.accel_bii
    Omega_BIb = inertial_data.Omega_BIb
    Q_i2b = inertial_data.Q_i2b
    Omeg_EIi = np.array([[0], [0], [0]])
    OmegDtI_EIi = np.array([[0], [0], [0]])
    Pos_bei, Vel_bEi, VelDtE_bEi, Omeg_BEb = _ECEF(pos_bii, vel_bIi, accel_bii, Omega_BIb, Q_i2b, Omeg_EIi, OmegDtI_EIi)
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
    ) = _local_ned_flattened(Pos_bei, Vel_bEi, VelDtE_bEi, Omeg_BEb, Q_i2b)
    AltMSL = AltGeod
    AltAGL = AltMSL
    AltPress = _pressure_altitude(env.Atmosphere.Pressure)
    Vel_bEb = quaternion_vector_transformation(Q_i2b, Vel_bEi).reshape((3, 1))
    VelDtE_bEb = quaternion_vector_transformation(Q_i2b, VelDtE_bEi).reshape((3, 1))
    chidot, gammadot = _flight_path_angle_rates(VelDtH_bEh, Vel_bEh)
    chi, gamma = _flight_path_angles(Vel_bEh)
    phi, theta, psi = _local_horizontal_to_321_euler(Q_h2b)
    euler = EulerAngles(phi=float(phi), theta=float(theta), psi=float(psi))
    Pos_beh_topo = _topodetic_potision(Pos_bee)
    latLonAlt = LatLonAlt(LatGeod=LatGeod, Lon=Lon, AltGeod=AltGeod)
    ipdb.set_trace()
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


def get_air_rel(world_rel: WorldRel, env: EnvData) -> AirRel:
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
    Vel_bWb, VelDtB_bWb, Omeg_BWb = _air_relative(
        Vel_tBb, VelDtB_tBb, Omeg_TBb, Vel_wEb, VelDtH_wEb, Vel_bEb, VelDtE_bEb, Omeg_BEb, Omeg_BHb
    )
    VelDtH_bVh, Vel_bVh = _air_real_in_ned_coordinates(Vel_bWb, VelDtB_bWb, Q_h2b, Omeg_BWb)

    Vtot = jnp.sqrt(jnp.sum(Vel_bWb**2))
    Veas = Vtot * (jnp.sqrt(Density / 0.0023768907688269184))
    Mach = Vtot / SpeedSound
    qbar = 0.5 * Density * (Vtot**2)
    qc = _supersonic(Mach, Static_Pres) if Mach > 1 else _subsonic(Mach, Static_Pres)
    Vcas = _calibrated_airspeed(qc)
    alpha = jnp.arctan2(Vel_bWb[2], Vel_bWb[0])
    beta = jnp.arctan2(Vel_bWb[1], jnp.sqrt(jnp.sum(jnp.array([Vel_bWb[0], Vel_bWb[2]]) ** 2)))
    chi, gamma, mu = _air_relative_chi_gamma_mu(Q_h2b, beta, alpha)
    alphadot, betadot, Vtotdot = _wind_angle_derivative(Vel_bWb, VelDtB_bWb)
    chidot, gammadot = _flight_path_angle_rates(VelDtH_bVh, Vel_bVh)
    mudot = _velocity_vector_bank_angle_rate(alpha, beta, gamma, chidot, alphadot, Omeg_BHb)
    phiAero = jnp.arctan2(Vel_bWb[1], Vel_bWb[2])
    alphaTotal = jnp.arctan2(jnp.sqrt(jnp.sum(Vel_bWb[1:] ** 2)), Vel_bWb[0])
    Q_h2v, Omeg_VHb = _velocity_vector_altitude_and_regular_rate(chi, gamma, chidot, gammadot, Q_h2b)
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


def _ECEF(
    pos_bii: Vec3_1,
    vel_bIi: Vec3_1,
    accel_bii: Vec3_1,
    Omega_BIb: Vec3_1,
    Q_i2b: Quat_1,
    Omeg_EIi: Vec3_1,
    OmegDtI_EIi: Vec3_1,
) -> tuple[Vec3_1, Vec3_1, Vec3_1, Vec3_1]:
    cross = jnp.cross(Omeg_EIi, vel_bIi, axis=0)
    VelDtE_bEi = accel_bii - jnp.cross(OmegDtI_EIi, pos_bii, axis=0) - cross * 2 + jnp.cross(Omeg_EIi, cross, axis=0)
    Vel_bEi = vel_bIi - cross
    Omeg_BEb = Omega_BIb - quaternion_vector_transformation(Q_i2b, Omeg_EIi).reshape((3, 1))
    Pos_bei = pos_bii
    return Pos_bei, Vel_bEi, VelDtE_bEi, Omeg_BEb


def _ECEF_noacc(
    pos_bii: Vec3_1,
    vel_bIi: Vec3_1,
    Omega_BIb: Vec3_1,
    Q_i2b: Quat_1,
    Omeg_EIi: Vec3_1,
) -> tuple[Vec3_1, Vec3_1, Vec3_1]:
    cross = jnp.cross(Omeg_EIi, vel_bIi, axis=0)
    Vel_bEi = vel_bIi - cross
    Omeg_BEb = Omega_BIb - quaternion_vector_transformation(Q_i2b, Omeg_EIi).reshape((3, 1))
    Pos_bei = pos_bii
    return Pos_bei, Vel_bEi, Omeg_BEb


def _get_local_ned_flattened_useful(
    Pos_bei: Vec3_1, Vel_bEi: Vec3_1, Omeg_BEb: Vec3_1, Q_i2b: Quat_1
) -> tuple[Quat_1, Vec3, Quat_1, FloatScalar]:
    Q_i2e: Quat_1 = np.array([[1], [0], [0], [0]])
    Pos_bee: Vec3_1 = quaternion_vector_transformation(Q_i2e, Pos_bei).reshape((3, 1))  # 1
    Lon, LatGeod, AltGeod = _flat_earth_to_geodetic(Pos_bee)

    CONSTANT_1 = np.array([[1], [0], [0], [0]])
    Q_h2i = quaternion_conjugate(CONSTANT_1)
    Q_h2b = quaternion_multiplication(Q_h2i, Q_i2b)

    Omeg_HEh = np.array([[0], [0], [0]])  # 5
    Omeg_BHb = Omeg_BEb - quaternion_vector_transformation(Q_h2b, Omeg_HEh).reshape((3, 1))  # 4

    # Negate to keep sign of scalar part of quaternion positive.
    Q_h2b = jnp.where(Q_h2b[0, 0] > 0, Q_h2b, -Q_h2b)
    # Q_h2b = Q_h2b if Q_h2b[0][0] > 0 else -Q_h2b  # Note: not sure if correct #7

    Vel_bEh = quaternion_vector_transformation(CONSTANT_1, Vel_bEi).reshape((3, 1))  # 2

    return Vel_bEh, Omeg_BHb, Q_h2b, AltGeod


def _local_ned_flattened(Pos_bei, Vel_bEi, VelDtE_bEi, Omeg_BEb, Q_i2b):
    Q_i2e = np.array([[1], [0], [0], [0]])
    Pos_bee = quaternion_vector_transformation(Q_i2e, Pos_bei).reshape((3, 1))  # 1
    Lon, LatGeod, AltGeod = _flat_earth_to_geodetic(Pos_bee)
    CONSTANT_1 = jnp.array([[1], [0], [0], [0]])
    Q_i2h = CONSTANT_1 if CONSTANT_1[0][0] > 0 else -CONSTANT_1  # Note: not sure if correct #6
    Q_h2i = quaternion_conjugate(CONSTANT_1)
    Q_h2b = quaternion_multiplication(Q_h2i, Q_i2b)
    Omeg_HEh = np.array([[0], [0], [0]])  # 5
    Omeg_BHb = Omeg_BEb - quaternion_vector_transformation(Q_h2b, Omeg_HEh).reshape((3, 1))  # 4
    Q_h2b = Q_h2b if Q_h2b[0][0] > 0 else -Q_h2b  # Note: not sure if correct #7
    VelDtE_bEh = quaternion_vector_transformation(CONSTANT_1, VelDtE_bEi).reshape((3, 1))
    Vel_bEh = quaternion_vector_transformation(CONSTANT_1, Vel_bEi).reshape((3, 1))  # 2
    VelDtH_bEh = VelDtE_bEh - jnp.cross(Omeg_HEh, Vel_bEh, axis=0)  # 3
    return Pos_bee, Vel_bEh, VelDtH_bEh, Omeg_BHb, Omeg_HEh, Q_i2h, Q_h2b, Lon, LatGeod, AltGeod


def _flat_earth_to_geodetic(Pos_bee: Vec3_1) -> tuple[FloatScalar, FloatScalar, FloatScalar]:
    assert Pos_bee.shape == (3, 1)
    altitude: Vec1 = -Pos_bee[2]
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
    u3: Vec1 = jnp.mod(lat_in - Min, Range) + Min
    u2: Vec1 = jnp.abs(u3)
    u1: Vec1 = (jnp.pi - u2) * jnp.sign(u3)

    latitude = jnp.where(u2 > jnp.pi / 2, u1, u3)
    assert latitude.shape == (1,)
    # if u2 > jnp.pi / 2:
    #     latitude = u1
    # else:
    #     latitude = u3

    u3_2 = long_in
    u2_2 = u2
    u1_2 = jnp.pi + long_in

    longitude = jnp.where(u2_2 > jnp.pi / 2, u1_2, u3_2)
    assert longitude.shape == (1,)
    # if u2_2 > jnp.pi / 2:
    #     longtitude = u1_2
    # else:
    #     longtitude = u3_2

    return longitude.squeeze(), latitude.squeeze(), altitude.squeeze()


def _pressure_altitude(p):
    UP = 1
    KP = 0.00047254133200895748
    EP = 0.19201354867458559
    KA = 145442.15626892794
    # KP = 0.00047254
    # EP = 0.19201
    # KA = 145440
    UA = 1
    return (1 - (p * UP * KP) ** (EP)) * KA * UA


def _topodetic_potision(Pos_bee):
    C = np.array([[0], [0], [0]])
    Q_e2h = np.array([[1], [0], [0], [0]])
    return quaternion_vector_transformation(Q_e2h, Pos_bee - C)


def _local_horizontal_to_321_euler(Q_h2b):
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


def _flight_path_angles(Vel_bEh):
    chi = jnp.arctan2(Vel_bEh[1], Vel_bEh[0])
    gamma = jnp.arctan2(-Vel_bEh[2], jnp.sqrt(jnp.sum(jnp.array([Vel_bEh[0], Vel_bEh[1]]) ** 2)))
    return chi, gamma


def _flight_path_angle_rates(VelDtH_bEh, Vel_bEh):
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


def _air_relative(Vel_tBb, VelDtB_tBb, Omeg_TBb, Vel_wEb, VelDtH_wEb, Vel_bEb, VelDtE_bEb, Omeg_BEb, Omeg_BHb):
    # pdb.set_trace()
    Vel_bWb = Vel_bEb - Vel_wEb - Vel_tBb
    VelDtB_bWb = (
        (VelDtE_bEb - jnp.cross(Omeg_BEb, Vel_bEb, axis=0))
        - (VelDtH_wEb - jnp.cross(Omeg_BHb, Vel_wEb, axis=0))
        - VelDtB_tBb
    )
    Omeg_BWb = Omeg_BHb - Omeg_TBb
    return Vel_bWb, VelDtB_bWb, Omeg_BWb


def _air_real_in_ned_coordinates(Vel_bWb, VelDtB_bWb, Q_h2b, Omeg_BWb):
    q = quaternion_conjugate(Q_h2b)
    # import pdb; pdb.set_trace()
    v_1 = VelDtB_bWb + jnp.cross(Omeg_BWb, Vel_bWb, axis=0)
    VelDtH_bVh = quaternion_vector_transformation(q, v_1)
    Vel_bVh = quaternion_vector_transformation(q, Vel_bWb)
    return VelDtH_bVh, Vel_bVh


def _supersonic(Mach, Static_Pres):
    Mach = Mach if Mach >= 1 else 1
    gamma = 1.4
    x_1 = Mach**2 * (gamma + 1) / 2
    u_1 = gamma / (gamma - 1)
    x_2 = Mach**2 * 2 * gamma / (gamma + 1) - (gamma - 1) / (gamma + 1)
    u_2 = 1 / (gamma - 1)
    qc = ((x_1 ** (u_1)) * (x_2 ** (u_2)) - 1) * Static_Pres
    return qc


def _subsonic(Mach, Static_Pres):
    gamma = 1.4
    x_1 = Mach**2 * (gamma - 1) / 2 + 1
    u_1 = gamma / (gamma - 1)
    qc = ((x_1 ** (u_1)) - 1) * Static_Pres
    return qc


def _calibrated_airspeed(qc):
    gamma = 1.4
    Pref = 2116.2166236739372
    rhoref = 0.0023768907688269184
    mul_1 = (qc / Pref + 1) ** ((gamma - 1) / gamma) - 1
    Vcas = jnp.sqrt(mul_1 * Pref / rhoref * (2 * gamma / (gamma - 1)))
    return Vcas


def _air_relative_chi_gamma_mu(Q_h2b, beta, alpha):
    q2 = quaternion_conjugate(
        quaternion_multiplication(single_axis_quaternion(-beta, 2), single_axis_quaternion(alpha, 1))
    )
    # pdb.set_trace()
    dcm = quaternion_to_DCM(quaternion_multiplication(Q_h2b, q2))
    chi = jnp.arctan2(dcm[0][1], dcm[0][0])
    gamma = jnp.arcsin(-dcm[0][2])
    mu = jnp.arctan2(dcm[1][2], dcm[2][2])
    return chi, gamma, mu


def _wind_angle_derivative(Vel_bVb, VelDtB_bVb):
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


def _velocity_vector_bank_angle_rate(alpha, beta, gamma, chidot, alphadot, Omeg_BHb):
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


def _velocity_vector_altitude_and_regular_rate(chi, gamma, chidot, gammadot, Q_h2b):
    Q_h2v = quaternion_multiplication(single_axis_quaternion(chi, 2), single_axis_quaternion(gamma, 1))
    Q_v2b = quaternion_multiplication(quaternion_conjugate(Q_h2v), Q_h2b)
    Omeg_NHn = jnp.array([chidot * -jnp.sin(gamma), gammadot, chidot * jnp.cos(gamma)])
    Omeg_VHb = quaternion_vector_transformation(Q_v2b, Omeg_NHn)
    return Q_h2v, Omeg_VHb
