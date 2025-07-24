import jax.numpy as jnp
import numpy as np

from jax_guam.guam_types import RefInputs
from jax_guam.utils.batch_spline import get_spline


def lift_cruise_reference_inputs(time):
    # return lift_cruise_reference_inputs_1(time)
    return lift_cruise_reference_inputs_2(time)


def lift_cruise_reference_inputs_1(time):
    timeseries = [0, 20, 40]
    vel_bIc_des = jnp.array([[0, 0, -8], [0, 0, 0], [15, 0, 0]])  # SimPar.RefInputs.SimInput.Vel_bIc_des
    pos_des = jnp.array([[0, 0, 0], [0, 0, -80], [150, 0, -100]])  # SimPar.RefInputs.SimInput.pos_des
    chi_des = jnp.array([0, 0, 0])  # SimPar.RefInputs.SimInput.chi_des
    chi_dot_des = jnp.array([0, 0, 0])  # SimPar.RefInputs.SimInput.chi_dot_des

    # Linear interpolate between the points.
    if time in timeseries:
        index = timeseries.index(time)
        return RefInputs(
            Vel_bIc_des=vel_bIc_des[index],
            Pos_des=pos_des[index],
            Chi_des=chi_des[index],
            Chi_dot_des=chi_dot_des[index],
        )
    elif time > timeseries[0] and time < timeseries[1]:
        first = (vel_bIc_des[1] - vel_bIc_des[0]) / (timeseries[1] - timeseries[0]) * time + vel_bIc_des[0]
        second = (pos_des[1] - pos_des[0]) / (timeseries[1] - timeseries[0]) * time + pos_des[0]
        return RefInputs(Vel_bIc_des=first, Pos_des=second, Chi_des=chi_des[0], Chi_dot_des=chi_dot_des[0])
    elif time > timeseries[1] and time < timeseries[2]:
        first = (vel_bIc_des[2] - vel_bIc_des[1]) / (timeseries[2] - timeseries[1]) * (
            time - timeseries[1]
        ) + vel_bIc_des[1]
        second = (pos_des[2] - pos_des[1]) / (timeseries[2] - timeseries[1]) * (time - timeseries[1]) + pos_des[1]
        return RefInputs(Vel_bIc_des=first, Pos_des=second, Chi_des=chi_des[0], Chi_dot_des=chi_dot_des[0])
    else:
        return RefInputs(Vel_bIc_des=np.zeros(3), Pos_des=pos_des[2], Chi_des=chi_des[2], Chi_dot_des=chi_dot_des[2])


def lift_cruise_reference_inputs_2(time: float):
    # First, hover. Then, go in a square.
    # (5, )
    T_t = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    # (5, 3)
    T_vel_bIc = np.array([[0, 0, -8], [20, 0, 0], [0, 20, 0], [-20, 0, 0], [0, 0, 0], [0, 0, 0]])
    # (5, 3)
    T_pos_bii = np.array([[0, 0, 0], [0, 0, -80], [200, 0, -80], [200, 200, -80], [0, 200, -80], [0, 200, -80]])

    T_scale = T_t[-1]
    spl_vel_bIc = get_spline(T_t / T_scale, T_vel_bIc, k=1, s=0)
    spl_pos_bii = get_spline(T_t / T_scale, T_pos_bii, k=1, s=0)

    vel_bIc = spl_vel_bIc(time / T_scale)
    pos_bii = spl_pos_bii(time / T_scale)

    assert vel_bIc.shape == (3,) and pos_bii.shape == (3,)

    return RefInputs(vel_bIc, pos_bii, Chi_des=np.array(0.0), Chi_dot_des=np.array(0.0))
