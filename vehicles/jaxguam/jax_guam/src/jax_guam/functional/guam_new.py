import functools as ft
from typing import NamedTuple

import jax
import tqdm

from jax_guam.functional.aero_prop_new import FuncAeroProp
from jax_guam.functional.lc_control import LCControl, LCControlState
from jax_guam.functional.surf_engine import SurfEngine, SurfEngineState
from jax_guam.functional.vehicle_eom_simple import VehicleEOMSimple
from jax_guam.guam_types import AircraftState, AircraftStateVec, EnvData, PwrCmd, RefInputs
from jax_guam.subsystems.environment.environment import Environment
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs
from jax_guam.subsystems.vehicle_model_ref.power_system import power_system
from jax_guam.utils.ode import ode3


class GuamState(NamedTuple):
    controller: LCControlState
    # [   0:3  ,   3:6    ,   6:9  ,  9:13 ]
    # [ vel_bEb, Omega_BIb, pos_bii, Q_i2b ]
    aircraft: AircraftStateVec
    surf_eng: SurfEngineState

    @property
    def pos_ned(self):
        return self.aircraft[..., 6:9]

    @staticmethod
    def create():
        return GuamState(
            controller=LCControlState.create(),
            aircraft=AircraftState.GetDefault13(),
            surf_eng=SurfEngineState.create(),
        )


class FuncGUAM:
    def __init__(self):
        self._environment = Environment()

        self.controller = LCControl()
        self.veh_eom = VehicleEOMSimple()
        self.surf_eng = SurfEngine()
        self.aero_prop = FuncAeroProp()

        self.dt = 0.005

    @property
    def env_data(self) -> EnvData:
        return self._environment.Env

    def deriv(self, state: GuamState, ref_inputs: RefInputs) -> GuamState:
        ref_inputs.assert_shapes()

        sensor, aeroprop_body_data, alt_msl = self.veh_eom.get_sensor_aeroprop_altmsl(state.aircraft)
        atmosphere = self._environment.get_env_atmosphere(alt_msl)
        env_data = self.env_data._replace(Atmosphere=atmosphere)
        control, cache = self.controller.get_control(state.controller, sensor, ref_inputs)

        d_state_controller = self.controller.state_deriv(cache)
        pwr_cmd = PwrCmd(CtrlSurfacePwr=control.Cmd.CtrlSurfacePwr, EnginePwr=control.Cmd.EnginePwr)
        power = power_system(pwr_cmd)

        surf_act, prop_act = self.surf_eng.get_surf_prop_act(state.surf_eng, control.Cmd, power)
        d_state_surf_eng = self.surf_eng.surf_engine_state_deriv(control.Cmd, state.surf_eng)

        fm = self.aero_prop.aero_prop(prop_act, surf_act, env_data, aeroprop_body_data)
        fm_total = self.veh_eom.get_fm_with_gravity(state.aircraft, fm)
        d_state_aircraft = self.veh_eom.state_deriv(fm_total, state.aircraft)

        return GuamState(d_state_controller, d_state_aircraft, d_state_surf_eng)

    def step(self, dt: float, state: GuamState, ref_inputs: RefInputs) -> GuamState:
        deriv_fn = ft.partial(self.deriv, ref_inputs=ref_inputs)
        state_new = ode3(deriv_fn, dt, state)
        # We also need to clip the state.
        state_new = state_new._replace(surf_eng=self.surf_eng.clip_state(state_new.surf_eng))
        return state_new

    def simulate(self, final_time: float = 10.0):
        state = GuamState.create()
        T_state = [state]
        T = int(final_time / self.dt)

        jit_step = jax.jit(self.step)

        for kk in tqdm.tqdm(range(T)):
            t = kk * self.dt
            ref_inputs = lift_cruise_reference_inputs(t)
            state = jit_step(self.dt, state, ref_inputs)
            T_state.append(state)

        return T_state
