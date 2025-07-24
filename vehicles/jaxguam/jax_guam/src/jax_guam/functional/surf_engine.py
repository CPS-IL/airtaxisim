import functools as ft
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from extensisq import BS5
from scipy.integrate import solve_ivp

from jax_guam.guam_types import Cmd, EngineControl, Failure_Engines, Failure_Surfaces, Power, PropAct, SurfAct, Trim
from jax_guam.utils.jax_types import BoolScalar, Vec5, Vec5_1, Vec9

# [flaperon L, flaperon R, elevator L, elevator R, Rudder]
CtrlSurfState = Vec5


class SurfEngineState(NamedTuple):
    ctrl_surf_state: CtrlSurfState

    @staticmethod
    def create() -> "SurfEngineState":
        return SurfEngineState(ctrl_surf_state=np.array([0, 0, -0.000780906088785921, -0.000780906088785921, 0]))


class SurfEngine:
    def __init__(self):
        # Constants:
        self.FREQ = np.array(
            [3.183098861837907, 3.183098861837907, 3.183098861837907, 3.183098861837907, 3.183098861837907]
        )
        self.RL = np.array([100, 100, 100, 100, 100])
        self.POS_HI = np.array(
            [0.52359877559829882, 0.52359877559829882, 0.52359877559829882, 0.52359877559829882, 0.52359877559829882]
        )
        self.POS_LO = -self.POS_HI

    @ft.partial(jax.jit, static_argnums=0)
    def get_surf_prop_act(self, state: SurfEngineState, cmd: Cmd, power: Power) -> tuple[SurfAct, PropAct]:
        surfaces = self._get_control_surfaces(state.ctrl_surf_state, cmd.CtrlSurfaceCmd)
        engines = self._get_propulsion(cmd.EngineCmd)
        return surfaces, engines

    @ft.partial(jax.jit, static_argnums=0)
    def surf_state_deriv(self, ctrl_surf_cmd: Vec5_1, state: CtrlSurfState) -> CtrlSurfState:
        """P controller."""
        assert ctrl_surf_cmd.shape == (5, 1)
        err = ctrl_surf_cmd.squeeze() - state
        assert err.shape == (5,)
        rate = err * 2 * jnp.pi * self.FREQ
        rate = rate.clip(-self.RL, self.RL)
        return rate

    def clip_state(self, state: SurfEngineState) -> SurfEngineState:
        """We need to clip the state to the limits."""
        ctrl_surf_state = state.ctrl_surf_state.clip(self.POS_LO, self.POS_HI)
        return SurfEngineState(ctrl_surf_state)

    def surf_engine_state_deriv(self, cmd: Cmd, state: SurfEngineState) -> SurfEngineState:
        ctrl_surf_state, _ = self.surf_state_deriv_saturate(cmd.CtrlSurfaceCmd, state.ctrl_surf_state)
        return SurfEngineState(ctrl_surf_state)

    def step_const_surf(self, cmd: Cmd, state: SurfEngineState, dt: float) -> SurfEngineState:
        def deriv_fn_wrapped(t, surf_state):
            return self.surf_state_deriv(cmd.CtrlSurfaceCmd, surf_state)

        sol = solve_ivp(deriv_fn_wrapped, [0.0, dt], state.ctrl_surf_state, t_eval=[dt], method=BS5)
        assert sol.y.shape == (5, 1)
        ctrl_surf_state = sol.y.squeeze(-1)
        assert ctrl_surf_state.shape == (5,)
        return self.clip_state(SurfEngineState(ctrl_surf_state))

    def surf_state_deriv_saturate(
        self, ctrl_surf_cmd: Vec5_1, state: CtrlSurfState
    ) -> tuple[CtrlSurfState, BoolScalar]:
        rate = self.surf_state_deriv(ctrl_surf_cmd, state)
        state_clipped = state.clip(self.POS_LO, self.POS_HI)
        is_saturated = state != state_clipped
        rate = jnp.where(is_saturated, 0, rate)
        return rate, is_saturated

    def _get_control_surfaces(self, ctrl_surf_state: CtrlSurfState, ctrl_surf_cmd: Vec5_1) -> SurfAct:
        assert ctrl_surf_state.shape == (5,)
        ctrl_surf_rate, is_saturated = self.surf_state_deriv_saturate(ctrl_surf_cmd, ctrl_surf_state)
        failure_surfaces = None
        return SurfAct(ctrl_surf_state, ctrl_surf_rate, failure_surfaces)

    def _get_propulsion(self, eng_cmd: EngineControl) -> PropAct:
        eng_speed = eng_cmd
        eng_accel = np.zeros(eng_cmd.shape)
        failure = None
        return PropAct(eng_speed, eng_accel, failure)
