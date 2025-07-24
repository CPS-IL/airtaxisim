import jax.numpy as jnp

from jax_guam.guam_types import EOM, EulerAngles, Sensor, SensorNoAcc
from jax_guam.utils.functions import nquaternion_to_Euler, quaternion_vector_transformation
from jax_guam.utils.jax_types import FloatScalar, Force, Mat3, Moment, Vec6, Vec13


def sensor_from_eom(eom: EOM) -> Sensor:
    Omeg_BIb = eom.InertialData.Omega_BIb
    Accel_bIi = eom.InertialData.accel_bii
    Q_i2b = eom.InertialData.Q_i2b
    Pos_bii = eom.InertialData.pos_bii
    Vel_bIi = eom.InertialData.vel_bIi
    LatLonAlt = eom.WorldRelativeData.LatLonAlt
    if eom.AirRelativeData is not None:
        Vtot = eom.AirRelativeData.Vtot
        gamma = eom.AirRelativeData.gamma
        chi = eom.AirRelativeData.chi
    else:
        Vtot = 0
        gamma = 0
        chi = 0
    Accel_bIb = quaternion_vector_transformation(Q_i2b, Accel_bIi).reshape((3, 1))
    phi, theta, psi = nquaternion_to_Euler(Q_i2b)

    return Sensor(
        Omeg_BIb=Omeg_BIb,
        Accel_bIb=Accel_bIb,
        Q_i2b=Q_i2b,
        Pos_bIi=Pos_bii,
        Vel_bIi=Vel_bIi,
        gpsLLA=jnp.array([[LatLonAlt.LatGeod], [LatLonAlt.Lon], [LatLonAlt.AltGeod]]),
        LaserAlt=LatLonAlt.AltGeod,
        Euler=EulerAngles(phi=float(phi), theta=float(theta), psi=float(psi)),
        Vtot=Vtot,
        gamma=gamma,
        chi=chi,
    )


def sensor_noacc_from_eom(eom: EOM) -> SensorNoAcc:
    Vel_bIi = eom.InertialData.vel_bIi
    Omeg_BIb = eom.InertialData.Omega_BIb
    Pos_bii = eom.InertialData.pos_bii
    Q_i2b = eom.InertialData.Q_i2b

    phi, theta, psi = nquaternion_to_Euler(Q_i2b)
    euler = EulerAngles(phi=phi, theta=theta, psi=psi)

    return SensorNoAcc(Vel_bIi, Omeg_BIb, Pos_bii, Q_i2b, euler)
