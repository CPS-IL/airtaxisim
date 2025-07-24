import jax.numpy as jnp

from jax_guam.classes.BodyClass import BodyClass
from jax_guam.classes.MassClass import MassClass
from jax_guam.classes.PropellerClass import PropellerClass
from jax_guam.classes.TailClass import TailClass
from jax_guam.classes.TiltWingClass import TiltWingClass
from jax_guam.classes.VerticalTailClass import VerticalTailClass
from jax_guam.classes.WingClass import WingClass
from jax_guam.classes.WingPropClass import WingPropClass
from jax_guam.data.read_data import read_data, read_data_dat
from jax_guam.utils.paths import data_dir


# translated from build_Lift_plus_Cruise.m
def init_tiltwing():
    # Fuselage
    m_b = 15.294862
    Ib = jnp.eye(3) * jnp.array([74.788921, 460.34866, 451.179883])
    body_cm_b = jnp.array([[-9], [0], [-4.25]])
    S_b = 42.07
    S_p = 172.69
    S_s = 368.851
    f_ld = 29.93 / 6.13

    # Create the Body
    Body = BodyClass(m_b, Ib, body_cm_b, S_b, S_p, S_s, f_ld)

    # Wing and Propeller
    wing_airfoil = jnp.array(read_data_dat(data_dir() / "naca633618.dat"))
    wing_aero_coeff = read_data(data_dir() / "NACA_0015_pp.mat")
    b = 47.5
    b_e = 47.5
    c = jnp.array([6.34513, 3.000])
    gamma = -2 * jnp.pi / 180
    m_w = 12.044978
    Iw = jnp.eye(3) * jnp.array([1608.52513, 12.672535, 1618.725065])
    w_cm_b = jnp.array([[-10.69], [0.000], [-8.5]])
    c4_b = w_cm_b

    y_flap = jnp.array([0.45, 0.95]) * b / 2
    y_aileron = jnp.array([0.45 * b / 2, 0.95 * b / 2])

    # Create the Wing
    Wing = WingClass(
        wing_airfoil, wing_aero_coeff["NACA_0015_pp"], [b, b_e], c, gamma, y_flap, y_aileron, c4_b, m_w, Iw, w_cm_b
    )

    # Create the Wing-Propeller Combination
    WingProp = WingPropClass(Wing)

    # Tail
    # Create the Horizontal Tail
    # NOTE: airfoil_t[:, 0] is all NaNs!
    airfoil_t = jnp.array(read_data_dat(data_dir() / "n0012.dat"))
    b_ht = 10.3328
    b_e_ht = 10.3328
    # Chord
    c_ht = jnp.array([3.00, 1.80])
    # Dihedral
    gamma = 0.0
    # Elevator location
    y_flap = jnp.array([0.0, 1.0]) * b_ht / 2
    # Aileron location (set to zero)
    y_aileron = jnp.array([0, 0])
    # Horizontal tail center of mass in the body frame
    ht_cm_b = jnp.array([[-29.1], [0.0], [-8.01]])
    # Horizontal tail quarter chord location in the body frame
    ht_c4_b = ht_cm_b
    # Mass of the horizontal tail
    m_ht = 2.262387
    # Inertia matrix for the horizontal tail
    Iht = jnp.eye(3) * jnp.array([3.473351, 8.689178, 5.338722])
    hTail = WingClass(
        airfoil_t,
        wing_aero_coeff["NACA_0015_pp"],
        [b_ht, b_e_ht],
        c_ht,
        gamma,
        y_flap,
        y_aileron,
        ht_c4_b,
        m_ht,
        Iht,
        ht_cm_b,
    )

    # Create the Vertical Tail
    # Span
    b_vt = 4.97524
    # Chord
    c_vt = jnp.array([8.5, 2.32902])
    # Vertical tail quarter chord location in the body frame
    vt_c4_b = jnp.array([[-28.35], [0.0], [-10.3]])
    # Mass of the vertical tail
    m_vt = 0.68167
    # Inertia matrix for the vertical tail
    Ivt = jnp.eye(3) * jnp.array([5.209589, 0.228829, 5.430544])
    # Vertical tail center of mass in the body frame
    vt_cm_b = vt_c4_b
    # Rudder location
    y_rudder = jnp.array([0, 1]) * b_vt
    vTail = VerticalTailClass(
        airfoil_t, wing_aero_coeff["NACA_0015_pp"], b_vt, c_vt, y_rudder, vt_c4_b, m_vt, Ivt, vt_cm_b
    )

    # Create the Tail with Horizontal and Vertical Components
    Tail = TailClass(hTail, vTail)

    # Lift and Cruise Propellers
    # Number of props
    NP = 9
    # Diameter of each propeller
    D = jnp.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0])
    # Propeller location in the body frame
    p_b = jnp.array(
        [
            [-5.07, -4.63, -4.63, -5.07, -19.2, -18.76, -18.76, -19.2, -31.94],
            [-18.75, -8.45, 8.45, 18.75, -18.75, -8.45, 8.45, 18.75, 0.0],
            [-6.73, -7.04, -7.04, -6.73, -9.01, -9.3, -9.3, -9.01, -7.79],
        ]
    )
    # Motor location in the body frame
    m_b = jnp.array(
        [
            [-5.07, -4.63, -4.63, -5.07, -19.2, -18.76, -18.76, -19.2, -31.94],
            [-18.75, -8.45, 8.45, 18.75, -18.75, -8.45, 8.45, 18.75, 0.0],
            [-7.6, -6.04, -6.04, -7.6, -8.0, -8.3, -8.3, -8.0, -7.79],
        ]
    )
    prop_coefs = read_data(data_dir() / "APCSF_10x4p7_coef.mat")
    # Propeller spin direction
    prop_spin = jnp.array([1, -1, 1, -1, -1, 1, -1, 1, 1])

    # Motor mass for each prop
    rotor_mass = 2.66855138  # slugs
    rotor_drive_sys_mass = 0.70223  # slugs
    rotor_engine_mass = 1.00984  # slugs
    rotor_assembly_mass = rotor_mass + rotor_drive_sys_mass + rotor_engine_mass

    pusher_mass = 2.129713  # slugs
    pusher_drive_sys_mass = 2.289138  # slugs
    pusher_engine_mass = 3.31803  # slugs
    pusher_assembly_mass = pusher_mass + pusher_drive_sys_mass + pusher_engine_mass

    # Create an array of motor masses
    m_m = jnp.concatenate([jnp.full(8, rotor_assembly_mass), jnp.array([pusher_assembly_mass])])
    p_T_e = jnp.array(
        [
            [0, 0.000, 0.000, 0, 0, 0.000, 0.000, 0, 1],
            [0, -0.139, 0.139, 0, 0, -0.139, 0.139, 0, 0],
            [-1, -0.990, -0.990, -1, -1, -0.990, -0.990, -1, 0],
        ]
    )

    # Create an array of props
    Props = []
    for ii in range(NP):
        prop = PropellerClass(
            prop_coefs["APCSF_10x4p7_coef"], prop_spin[ii], D[ii], p_b[:, ii], m_b[:, ii], m_m[ii], p_T_e[:, ii]
        )
        Props.append(prop)

    # Rotor Booms Masses
    # Pusher Prop Engine
    # mass
    pusher_drive_sys_mass = 1.7809
    pusher_eng_mass = 13.5389
    pusher_eng_asbly_mass = pusher_drive_sys_mass + pusher_eng_mass
    # inertia matrix
    pusher_eng_I = jnp.eye(3) * jnp.array([0.000, 0.000, 0.000])
    # location in the body frame
    pusher_eng_cm_b = jnp.array([[-30.94], [0.00], [-7.79]])
    Pusher_Eng = MassClass(pusher_eng_asbly_mass, pusher_eng_I, pusher_eng_cm_b)

    # Pusher Prop Generator
    # mass
    pusher_gen_mass = 4.2643
    # inertia matrix
    pusher_gen_I = jnp.eye(3) * jnp.array(jnp.array([0.000, 0.000, 0.000]))
    # location in the body frame
    pusher_gen_cm_b = jnp.array([[-30.94], [0.00], [-7.79]])
    Pusher_Gen = MassClass(pusher_gen_mass, pusher_gen_I, pusher_gen_cm_b)

    # Landing Gear
    # mass
    landing_gear_mass = 8.48397  # See Ref. [2] cells B59-B61
    # inertia matrix
    landing_gear_I = jnp.eye(3) * jnp.array(jnp.array([0.000, 0.000, 0.000]))
    # location in the body frame
    landing_gear_cm_b = jnp.array(
        [[-13.32464336], [-3.0924e-06], [-1.391179415]]
    )  # See Ref. [2], computed gear CM using cells B59-E61 & B122=E124
    Landing_Gear = MassClass(landing_gear_mass, landing_gear_I, landing_gear_cm_b)

    # Fuel Tank 1
    # mass
    fuel_tank_1_mass = 7.9325  # See Ref. [2], sum cells B35+B36
    fuel_liq_1_mass = 0.0
    fuel_1_mass = fuel_tank_1_mass + fuel_liq_1_mass
    # inertia matrix
    fuel_1_I = jnp.eye(3) * jnp.array(jnp.array([0.000, 0.000, 0.000]))
    # location in the body frame
    fuel_1_cm_b = jnp.array([[-12.773578], [0.00], [-3.130658]])  # See Ref. [2] cells C35-E36
    Fuel_1 = MassClass(fuel_1_mass, fuel_1_I, fuel_1_cm_b)

    # Fuel Tank 2
    # mass
    fuel_tank_2_mass = 1.899172  # See Ref. [2] cell B127
    fuel_liq_2_mass = 0.0
    fuel_2_mass = fuel_tank_2_mass + fuel_liq_2_mass
    # inertia matrix
    fuel_2_I = jnp.eye(3) * jnp.array(jnp.array([0.000, 0.000, 0.000]))
    # location in the body frame
    fuel_2_cm_b = jnp.array([[-2.75], [0.00], [-4.15]])  # See Ref. [2] cell C127-E127
    Fuel_2 = MassClass(fuel_2_mass, fuel_2_I, fuel_2_cm_b)

    # Systems
    sys_mass = 16.223758  # See Ref. [2] cell B126
    sys_I = jnp.eye(3) * jnp.array(jnp.array([0.0, 0.0, 0.0]))
    sys_cm_b = jnp.array([[-16], [0], [-6]])  # See Ref. [2] cells C126-E126
    Sys = MassClass(sys_mass, sys_I, sys_cm_b)

    # Systems
    ext_mass = 54.60  # Remainder of mass (Estimated)
    ext_I = jnp.eye(3) * jnp.array(jnp.array([0.0, 0.0, 0.0]))
    ext_cm_b = jnp.array([[-7.6], [0], [-0.5]])  # Notional location (Estimated)
    Ext = MassClass(ext_mass, ext_I, ext_cm_b)

    Extra_Mass = [Pusher_Eng, Pusher_Gen, Landing_Gear, Fuel_1, Fuel_2, Sys, Ext]

    # Tiltwing Aircraft
    tiltwing = TiltWingClass(WingProp, Tail, Body, Props, Extra_Mass)

    return tiltwing
