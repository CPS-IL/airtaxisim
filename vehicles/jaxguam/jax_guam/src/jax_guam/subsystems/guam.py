# import pdb
# import time
#
# import ipdb
# import jax.numpy as jnp
# import numpy as np
# import tqdm
# from loguru import logger
#
# from jax_guam.functional.aero_prop_new import FuncAeroProp
# from jax_guam.functional.lc_control import LCControl, LCControlState
# from jax_guam.functional.surf_engine import SurfEngine, SurfEngineState
# from jax_guam.functional.vehicle_eom_simple import VehicleEOMSimple
# from jax_guam.guam_types import *
# from jax_guam.guam_types import RefInputs, SimOutputs, TrimInputs
# from jax_guam.subsystems.environment.environment import Environment
# from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs
# from jax_guam.subsystems.vehicle_generalized_control.gen_ctrl import L_C_control
# from jax_guam.subsystems.vehicle_model_ref.aero_prop import AeroProp
# from jax_guam.subsystems.vehicle_model_ref.gear_system import gear_system
# from jax_guam.subsystems.vehicle_model_ref.power_system import power_system
# from jax_guam.subsystems.vehicle_model_ref.sensors import sensor_from_eom, sensor_noacc_from_eom
# from jax_guam.subsystems.vehicle_model_ref.surf_eng import SurfEng
# from jax_guam.subsystems.vehicle_model_ref.vehicle_eom_simp import VehicleEOMRef
# from jax_guam.utils.jax_utils import tree_mac
#
#
# class GUAM:
#     veh_eom: VehicleEOMSimple | VehicleEOMRef
#
#     def __init__(self):
#         # self.Pos_bii = jnp.zeros((3, 1))
#         # self.Sensor = Sensor(
#         #     Omeg_BIb=jnp.zeros((3, 1)),
#         #     Accel_bIb=jnp.zeros((3, 1)),
#         #     Q_i2b=jnp.zeros((4, 1)),
#         #     Pos_bIi=jnp.zeros((3, 1)),
#         #     Vel_bIi=jnp.zeros((3, 1)),
#         #     gpsLLA=jnp.zeros((3, 1)),
#         #     LaserAlt=0,
#         #     Euler=Euler(phi=0, theta=0, psi=0),
#         #     Vtot=0,
#         #     gamma=0,
#         #     chi=0,
#         # )
#
#         # self.EOM = None
#         self.time = 0
#         self.environment = Environment()
#
#         # self.controller = L_C_control()
#         self.controller = LCControl()
#         self.controller_state = LCControlState.create()
#
#         self.NEW_EOM = True
#         if self.NEW_EOM:
#             self.veh_eom = VehicleEOMSimple()
#         else:
#             self.veh_eom = VehicleEOMRef()
#
#         self.veh_eom_state = AircraftState.GetDefault13()
#
#         # self.surf_eng = SurfEng()
#         self.surf_eng = SurfEngine()
#         self.surf_eng_state = SurfEngineState.create()
#
#         # self.aero_prop = AeroProp()
#         self.aero_prop = FuncAeroProp()
#         self.dt = 0.005
#
#     def guam(self, trimInputs, final_time: float = 10.0):
#         T_output = []
#         T = int(final_time / self.dt)
#         for _ in tqdm.tqdm(range(T)):
#             # for _ in range(T):
#             #     print(self.time)
#             refInputs = lift_cruise_reference_inputs(self.time)
#             # print(refInputs)
#             # import pdb; pdb.set_trace()
#             trimInputs = TrimInputs(Engines=trimInputs[:9], Surfaces=trimInputs[9:])
#             SimOutputs = self.vehicle_simulation(refInputs, trimInputs)
#             # import pdb; pdb.set_trace()
#             # eom = SimOutputs.Vehicle.EOM
#             # self.Pos_bii = eom.InertialData.pos_bii
#             T_output.append(SimOutputs)
#             self.time += self.dt
#         return T_output
#
#     def vehicle_simulation(self, refInputs: RefInputs, trimInputs: TrimInputs) -> SimOutputs:
#         t0 = time.time()
#         # Note: This uses memory_tas and memory_AltMSL to update Wind and Turbulence.
#         self.environment.environment_turbulence_wind()
#         if isinstance(self.veh_eom, VehicleEOMSimple):
#             sensor, aeroprop_body_data, alt_msl = self.veh_eom.get_sensor_aeroprop_altmsl(self.veh_eom_state)
#         else:
#             assert isinstance(self.veh_eom, VehicleEOMRef)
#             eom = self.veh_eom.vehicle_eom_init(derivs=None)
#
#             alt_msl = eom.WorldRelativeData.AltMSL
#
#             aeroprop_body_data = eom.get_aeroprop_body_data()
#             # sensor = sensor_from_eom(eom)
#             sensor = sensor_noacc_from_eom(eom)
#
#         atmosphere = self.environment.get_env_atmosphere(alt_msl)
#         self.environment.set_env_atmosphere(atmosphere)
#
#         # self.environment.environment_atmosphere(eom)
#         # <-- 0.09
#         t1 = time.time()
#
#         # control = self.controller.lift_cruise_control(self.time, self.Sensor, refInputs)
#
#         controller_state = self.controller_state
#         control, cache = self.controller.get_control(controller_state, sensor, refInputs)
#         self.controller_state = tree_mac(self.controller_state, self.dt, self.controller.state_deriv(cache))
#
#         # np.set_printoptions(linewidth=300)
#         # logger.info("{} | {}".format(control.Cmd.EngineCmd.squeeze(), control.Cmd.CtrlSurfaceCmd.squeeze()))
#
#         # pos = self.VehicleEOMRef.EOM.InertialData.pos_bii
#         # np.set_printoptions(linewidth=280)
#         # logger.info("pos: {}\n{} | {}".format(pos.squeeze(), control.Cmd.EngineCmd.squeeze(), control.Cmd.CtrlSurfaceCmd.squeeze()))
#         # ipdb.set_trace()
#         # <-- 0.07
#         t2 = time.time()
#         pwrcmd = PwrCmd(CtrlSurfacePwr=control.Cmd.CtrlSurfacePwr, EnginePwr=control.Cmd.EnginePwr)
#         Power = power_system(pwrcmd)
#         # <-- 0.00
#         t3 = time.time()
#
#         if isinstance(self.surf_eng, SurfEngine):
#             surf_act, prop_act = self.surf_eng.get_surf_prop_act(self.surf_eng_state, control.Cmd, Power)
#             self.surf_eng_state = self.surf_eng.step_const_surf(control.Cmd, self.surf_eng_state, self.dt)
#         else:
#             assert isinstance(self.surf_eng, SurfEng)
#             surf_act, prop_act = self.surf_eng.surf_eng(self.time, control.Cmd, Power)
#
#         # <-- 0.013
#         t4 = time.time()
#         fm = self.aero_prop.aero_prop(prop_act, surf_act, self.environment.Env, aeroprop_body_data)
#         # <-- 0.001
#         t5 = time.time()
#
#         # logger.info("Total Moments: {}".format(self.FM.TotalFM.Moments_b.flatten()))
#
#         if isinstance(self.veh_eom, VehicleEOMSimple):
#             self.veh_eom_state = self.veh_eom.step_const_fm(fm, self.veh_eom_state, self.dt)
#             eom_new = self.veh_eom.get_eom(self.veh_eom_state, fm, self.environment.Env)
#         else:
#             eom_new = self.veh_eom.equ_motion(self.time, fm, self.environment.Env)
#
#         # <-- 0.288
#         t6 = time.time()
#         # sensor_new = sensor_from_eom(eom_new)
#         sensor_new = None
#         # # Note: This updates memory_tas and memory_AltMSL, to be used in the next step.
#         # self.environment.environment_param_update(eom_new, self.environment.Env)
#
#         GearCmd = control.Cmd.GearCmd
#         Gear = gear_system(GearCmd)
#         vehicle_out = VehicleOut(
#             Power=Power,
#             SurfAct=SurfAct,
#             PropAct=PropAct,
#             Gear=Gear,
#             FM=fm,
#             EOM=eom_new,
#             Sensor=sensor_new,
#         )
#         t7 = time.time()
#         # logger.info(
#         #     "t1-t0: {:.3f}, t2-t1: {:.3f}, t3-t2: {:.3f}, t4-t3: {:.3f}, t5-t4: {:.3f}, t6-t5: {:.3f}, t7-t6: {:.3f}".format(
#         #         t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6
#         #     )
#         # )
#         return SimOutputs(
#             Env=self.environment.Env, Vehicle=vehicle_out, Control=control, RefInputs=refInputs, Time=self.time
#         )
