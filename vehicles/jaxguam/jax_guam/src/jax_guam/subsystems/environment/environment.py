import jax.numpy as jnp
from jax import random

from jax_guam.guam_types import EOM, Atmosphere, EnvData, Turbulence, Wind
from jax_guam.utils.functions import lookup, quaternion_vector_transformation
from jax_guam.utils.jax_types import FloatScalar, Vec6


class Environment:
    def __init__(self):
        self.memory_tas = 70
        self.memory_AltMSL = 1000
        self.memory_mul = 1
        self.memory_mul2 = 1
        self.memory_mul3 = 1
        self.memory_mul4 = 1
        self.memory_mul5 = 1
        self.memory_mul6 = 1
        self.memory_mul7 = 1
        self.memory_mul8 = 1
        self.memory_mul9 = 1
        self.memory_Large_step_warning = 0
        self.key1 = random.PRNGKey(3887)
        self.key2 = random.PRNGKey(544)
        self.key3 = random.PRNGKey(1890)
        self.wind = Wind(Vel_wHh=jnp.zeros((3, 1)), Vel_DtH_wHh=jnp.zeros((3, 1)))
        self.turbulence = Turbulence(
            Vel_tBb=jnp.zeros((3, 1)),
            VelDtB_tBb=jnp.zeros((3, 1)),
            Omeg_TBb=jnp.zeros((3, 1)),
            OmegDtB_TBb=jnp.zeros((3, 1)),
        )
        self.atmosphere = Atmosphere(Density=0.0024, Pressure=2.1162e03, Temperature=518.6700, SpeedOfSound=1.4)  # TODO
        self.Env = EnvData(Wind=self.wind, Turbulence=self.turbulence, Atmosphere=self.atmosphere)

    def environment_turbulence_wind(self):
        tas = self.memory_tas
        AltMSL = self.memory_AltMSL
        level = jnp.array([0, 0, 0, 0])  # Note level select
        u_gust, v_gust, w_gust = self.turbulence_model(tas, AltMSL, level[0])
        # Vel_tBb = jnp.array([[u_gust*level[1]], [v_gust*level[2]], [w_gust*level[3]]])
        Vel_tBb = jnp.array([[0], [0], [0]])
        self.turbulence = Turbulence(
            Vel_tBb=Vel_tBb * 3.2808,
            VelDtB_tBb=jnp.array([[0], [0], [0]]),
            Omeg_TBb=jnp.array([[0], [0], [0]]),
            OmegDtB_TBb=jnp.array([[0], [0], [0]]),
        )
        mul1 = 10 * (-1.6878)
        mul2 = 0 * jnp.pi / 180
        Vel_wHh = jnp.array([mul1 * jnp.cos(mul2), mul1 * jnp.sin(mul2), 0])
        self.wind = Wind(Vel_wHh=Vel_wHh, Vel_DtH_wHh=jnp.array([[0], [0], [0]]))
        self.Env = EnvData(Wind=self.wind, Turbulence=self.turbulence, Atmosphere=self.atmosphere)

    def get_env_atmosphere(self, AltMSL: FloatScalar) -> Atmosphere:
        dt = 0
        atmos = atmos76(AltMSL, dt)
        atmosphere = Atmosphere(Density=atmos[0], Pressure=atmos[1], Temperature=atmos[2], SpeedOfSound=atmos[3])
        return atmosphere

    def set_env_atmosphere(self, atmosphere: Atmosphere) -> None:
        self.Env = EnvData(Wind=self.wind, Turbulence=self.turbulence, Atmosphere=atmosphere)
        self.atmosphere = atmosphere

    def environment_atmosphere(self, eom: EOM):
        dt = 0
        AltMSL = eom.WorldRelativeData.AltMSL
        atmos = atmos76(AltMSL, dt)
        self.atmosphere = Atmosphere(
            Density=float(atmos[0]), Pressure=float(atmos[1]), Temperature=float(atmos[2]), SpeedOfSound=float(atmos[3])
        )
        self.Env = EnvData(Wind=self.wind, Turbulence=self.turbulence, Atmosphere=self.atmosphere)

    def environment_param_update(self, eom: EOM, Wind_b: EnvData) -> EnvData:
        Vel_bEb = eom.WorldRelativeData.Vel_bEb
        Q_h2b = eom.WorldRelativeData.Q_h2b
        Vel_wHh = Wind_b.Wind.Vel_wHh
        mul = quaternion_vector_transformation(Q_h2b, Vel_wHh).reshape((3, 1))
        dot = (Vel_bEb - mul).T.dot(Vel_bEb - mul)
        gain = 0.592487
        self.memory_tas = jnp.sqrt(abs(dot)) * gain if dot > 0 else -jnp.sqrt(abs(dot)) * gain
        self.memory_AltMSL = eom.WorldRelativeData.AltMSL

    def turbulence_model(self, tas, alt, switch):
        vel = 0.514444444 * max(tas, 0.01)
        alt = 0.3048 * max(alt, 0.01)
        selector = 1
        intensity = 3
        u_gust, v_gust, w_gust, Large_step_warning = self.dryden_model(vel, alt, selector, intensity)
        return u_gust * 3.2808399, v_gust * 3.2808399, w_gust * 3.2808399

    def dryden_model(self, vel, alt, selector, intensity):
        # Note display not implemented
        u_gust, RMS_input, Longitudinal_step_warning = self.longitudinal_turbulence(selector, intensity, vel, alt)
        v_gust, RMS_input_2, Lateral_step_warning = self.lateral_turbulence(selector, intensity, vel, alt)
        w_gust, RMS_input_3, Vertical_step_warning = self.vertical_turbulence(selector, intensity, vel, alt)
        compare = (Longitudinal_step_warning + Lateral_step_warning + Vertical_step_warning) >= 1
        self.memory_Large_step_warning += compare
        return u_gust, v_gust, w_gust, self.memory_Large_step_warning

    def longitudinal_turbulence(self, selector, intensity, vel, alt):
        dt = 0.005
        noise = random.uniform(key=self.key1) * dt  # TODO: need to generate?
        RMS_input = self.filter_parameters(selector, intensity, alt)
        mul = noise * RMS_input[0] * jnp.sqrt(3.141593 / dt)
        mul = mul - self.memory_mul * vel / RMS_input[1]
        self.memory_mul = 1 / mul
        u_gust = self.memory_mul * jnp.sqrt(2 / 3.141593 * vel / RMS_input[1])
        Large_step_warnnig_flag = vel / RMS_input[1] * 0.005 > 0.11
        return u_gust, RMS_input[0], Large_step_warnnig_flag

    def lateral_turbulence(self, selector, intensity, vel, alt):
        dt = 0.005
        noise = random.uniform(key=self.key1) * dt  # TODO: need to generate?
        RMS_input = self.filter_parameters(selector, intensity, alt)
        mul = noise * RMS_input[0] * jnp.sqrt(3.141593 / dt)
        out = mul - self.memory_mul6 * vel / RMS_input[1]
        self.memory_mul6 = 1 / out
        out = self.memory_mul6 - self.memory_mul7 * vel / RMS_input[1]
        self.memory_mul7 = 1 / out
        plus_1 = 3 ** (0.5) * self.memory_mul7 * vel / RMS_input[1]

        out = mul - self.memory_mul8 * vel / RMS_input[1]
        self.memory_mul8 = 1 / out
        out = self.memory_mul8 - self.memory_mul9 * vel / RMS_input[1]
        self.memory_mul9 = 1 / out
        mult_2 = plus_1 + self.memory_mul9
        w_gust = mult_2 * jnp.sqrt(3 / 3.141593 * vel / RMS_input[1])

        Large_step_warnnig_flag = vel / RMS_input[1] * 0.005 > 0.11
        return w_gust, RMS_input[0], Large_step_warnnig_flag

    def vertical_turbulence(self, selector, intensity, vel, alt):
        dt = 0.005
        noise = random.uniform(key=self.key1) * dt  # TODO: need to generate?
        RMS_input = self.filter_parameters(selector, intensity, alt)
        mul = noise * RMS_input[0] * jnp.sqrt(3.141593 / dt)
        out = mul - self.memory_mul2 * vel / RMS_input[1]
        self.memory_mul2 = 1 / out
        out = self.memory_mul2 - self.memory_mul3 * vel / RMS_input[1]
        self.memory_mul3 = 1 / out
        plus_1 = 3 ** (0.5) * self.memory_mul3 * vel / RMS_input[1]

        out = mul - self.memory_mul4 * vel / RMS_input[1]
        self.memory_mul4 = 1 / out
        out = self.memory_mul4 - self.memory_mul5 * vel / RMS_input[1]
        self.memory_mul5 = 1 / out
        mult_2 = plus_1 + self.memory_mul5
        v_gust = mult_2 * jnp.sqrt(3 / 3.141593 * vel / RMS_input[1])

        Large_step_warnnig_flag = vel / RMS_input[1] * 0.005 > 0.11
        return v_gust, RMS_input[0], Large_step_warnnig_flag

    def filter_parameters(self, selector, intensity, alt):
        # Tm-4511 severe data
        input_values = [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            200,
            304.8,
            305,
            400,
            762,
            769,
            1000,
            2000,
            4000,
            6000,
            8000,
            10000,
            12000,
            14000,
            16000,
        ]
        if intensity == 1:
            table_data = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.17,
                0.17,
                0.2,
                0.21,
                0.22,
                0.22,
                0.25,
                0.26,
                0.24,
            ]
        elif intensity == 2:
            table_data = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1.65,
                1.65,
                2.04,
                2.13,
                2.15,
                2.23,
                2.47,
                2.62,
                2.44,
            ]
        else:
            table_data = [
                2.31,
                2.58,
                2.75,
                2.88,
                2.98,
                3.07,
                3.15,
                3.22,
                3.28,
                3.33,
                3.72,
                3.95,
                4.37,
                4.39,
                4.39,
                5.7,
                5.7,
                5.8,
                6.24,
                7.16,
                7.59,
                7.72,
                7.89,
                6.93,
                5,
            ]
        table_data_2 = [
            21,
            21,
            33,
            43,
            52,
            61,
            68,
            75,
            82,
            89,
            95,
            149,
            196,
            300,
            300,
            533,
            533,
            832,
            902,
            1040,
            1040,
            1040,
            1230,
            1800,
            2820,
            3400,
        ]
        sigma = lookup(alt, input_values, table_data)
        scale_length = lookup(alt, input_values, table_data_2)

        # MIL-STD-1797A APPENDIX A
        if intensity == 1:
            input_values = [2000, 9000, 18000, 100000]
            table_data = [5, 5, 3, 3]
        elif intensity == 2:
            input_values = [2000, 12000, 44000, 100000]
            table_data = [10, 10, 3, 3]
        else:
            input_values = [2000, 5000, 22000, 80000, 100000]
            table_data = [15.5, 21.5, 21.5, 3, 3]
        alt = alt * 1 / 0.3048
        # import pdb; pdb.set_trace()
        selection1 = jnp.array([lookup(alt, input_values, table_data), 1750])

        mul1 = intensity * 2.5
        mul2 = jnp.log(max(alt, 0.15) / 0.15) / jnp.log(20 / 0.15)
        mul3 = max(1, (0.177 + 0.000823 * alt) ** (-0.4))
        sec = alt / (0.177 + 0.000823 * alt) ** (1.2)
        selection2 = jnp.array([mul1 * mul2 * mul3, sec])

        out = selection1 if alt >= 2000 else selection2
        out = out * 0.3048
        sigma_2 = out[0]
        scale_length_2 = out[1]

        if selector == 1:
            return jnp.array([sigma, scale_length])
        else:
            return jnp.array([sigma_2, scale_length_2])
        # Note, display did not implemented


def atmos76(z: FloatScalar, dt: float) -> Vec6:
    # Constants from 1976 std. atmosphere
    r = 1716.55915670803
    gamma = 1.4
    g = 32.17405
    re = 20855531.5
    tsl = 288.15 * 1.8
    psl = 2116.22
    hsl = 0.0
    sldens = 0.0023769  # Atmospheric density at sea level
    slpr = 2116.220  # Sea level pressure, psf

    # Conversion from metric lapse rates to english lapse rates.
    # meters / 0.3048 = feet
    # kelvin * 1.8 = rankine
    rm2ft = 1000.0 / 0.3048
    conlapse = 1.8 / rm2ft

    # Convert geometric alt. to geopotential alt.
    h = (re * z) / (re + z)

    # Ensure we are in layer one (based on geopotential alt)
    # if h > 11 * rm2ft:
    #     h = 11 * rm2ft
    # if h < 0:
    #     h = 0
    h = h.clip(0.0, 11 * rm2ft)

    a1 = -6.5 * conlapse
    t1 = tsl + a1 * (h - hsl)
    del1 = (t1 / tsl) ** (-g / r / a1)
    p1 = del1 * psl
    t1 = t1 + dt
    rho1 = p1 / (r * t1)
    # import pdb; pdb.set_trace() # z gets veryvery large need to check input
    c1 = jnp.sqrt(gamma * (r * t1))
    return jnp.array([rho1, p1, t1 / 1.8, c1, p1 / slpr, jnp.sqrt(rho1 / sldens)])
