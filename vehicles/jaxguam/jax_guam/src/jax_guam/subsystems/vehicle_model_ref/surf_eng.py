import functools as ft

import ipdb
import jax.numpy as jnp
from scipy.integrate import solve_ivp

from jax_guam.guam_types import Cmd, Failure_Engines, Failure_Surfaces, Power, PropAct, SurfAct, Trim
from jax_guam.utils.jax_types import Vec5, Vec9, Vec5_1


class SurfEng:
    def __init__(self):
        self.pos_ctrlsurf = jnp.array([0, 0, -0.000780906088785921, -0.000780906088785921, 0])
        self.initial_pos_ctrlsurf = jnp.array([0, 0, -0.000780906088785921, -0.000780906088785921, 0])

        # Note: 1 if saturated, 0 if not saturated. Used as otpuut for the current rate.
        self.mul = jnp.ones(5)

        # Constants:
        self.FREQ = jnp.array(
            [3.183098861837907, 3.183098861837907, 3.183098861837907, 3.183098861837907, 3.183098861837907]
        )
        self.RL = jnp.array([100, 100, 100, 100, 100])
        self.POS_HI = jnp.array(
            [0.52359877559829882, 0.52359877559829882, 0.52359877559829882, 0.52359877559829882, 0.52359877559829882]
        )
        self.POS_LO = -self.POS_HI

    def surf_eng(self, time, cmd: Cmd, power: Power):
        surfaces = self._control_surfaces(time, cmd, power)
        engines = self._propulsion(cmd, power)
        return surfaces, engines

    def _control_surfaces(self, time, cmd: Cmd, power: Power) -> SurfAct:
        surf_cmd = cmd.CtrlSurfaceCmd
        surf_pwr = power.CtrlSurfacePwr
        # ctl_surf_dyn
        surfact = self._ctl_surf_dyn(time, surf_cmd, surf_pwr)
        return surfact

    def _ctl_surf_dyn(self, time, surf_cmd: Vec5_1, surf_pwr: Vec5) -> SurfAct:
        surfact = self._first_order_actuator(time, surf_cmd)
        return surfact

    def _first_order_actuator(self, time, ctrl_surf_cmd: Vec5_1) -> SurfAct:
        # questions, integral? is cmd a function of t? first order function multiple blocks in the middle? clipping?
        y_dot_lim = self._first_order_actuator_integral(time, ctrl_surf_cmd, self.pos_ctrlsurf)
        rate = (y_dot_lim * self.mul).reshape((5, 1))
        surf_failure = Failure_Surfaces(
            F_Fail_Initiate=jnp.zeros(5),
            F_Hold_Last=jnp.zeros(5),
            F_Pre_Scale=jnp.ones(5),
            F_Post_Scale=jnp.ones(5),
            F_Pos_Bias=jnp.zeros(5),
            F_Pos_Scale=jnp.ones(5),
            F_Up_Plim=self.POS_HI,
            F_Lwr_Plim=self.POS_LO,
            F_Rate_Bias=jnp.zeros(5),
            F_Rate_Scale=jnp.ones(5),
            F_Up_Rlim=self.RL,
            F_Lwr_Rlim=-self.RL,
            F_Accel_Bias=jnp.zeros(5),
            F_Accel_Scale=jnp.ones(5),
            F_Up_Alim=jnp.zeros(5) + jnp.inf,
            F_Lwr_Alim=jnp.zeros(5) - jnp.inf,
            F_Gen_Sig_Sel=jnp.zeros(15),
        )
        surfact = SurfAct(CtrlSurfPos=self.pos_ctrlsurf, CtrlSurfRate=rate, Failure_Surfaces=surf_failure)
        ##################################################################
        # Update internal state.
        pos_ctrlsurf = self.pos_ctrlsurf

        deriv_fn = ft.partial(self._first_order_actuator_integral, ctrl_surf_cmd)
        sol = solve_ivp(deriv_fn, [time, time + 0.005], pos_ctrlsurf, t_eval=[time + 0.005])
        y = sol.y.reshape(5)
        y_clipped = jnp.clip(y, self.POS_LO, self.POS_HI)
        not_saturated = y_clipped == y
        self.pos_ctrlsurf = y_clipped
        # If
        self.mul = jnp.ones(5) * not_saturated
        ##################################################################
        return surfact

    def _first_order_actuator_integral(self, ctrl_surf_cmd, time, pos_ctrlsurf):
        y_dot = (
            (ctrl_surf_cmd.reshape(5) - pos_ctrlsurf) * 2 * jnp.pi * self.FREQ
        )  # check if it is elementwise multiplication
        y_dot_lim = jnp.clip(y_dot, -self.RL, self.RL)  # clip this to be RL and -RL, not sure if need divide
        return y_dot_lim

    def _propulsion(self, cmd, power):
        eng_cmd = cmd.EngineCmd
        eng_pwr = power.EnginePwr
        prop_act = self._prop_dyn(eng_cmd, eng_pwr)
        return prop_act

    def _prop_dyn(self, eng_cmd, eng_pwr):
        prop_act = self._none_eng(eng_cmd, eng_pwr)
        return prop_act

    def _none_eng(self, eng_cmd, eng_pwr):
        eng_speed = eng_cmd
        eng_accel = eng_cmd * 0
        eng_failure = surf_failure = Failure_Engines(
            F_Fail_Initiate=jnp.zeros(9),
            F_Hold_Last=jnp.zeros(9),
            F_Pre_Scale=jnp.ones(9),
            F_Post_Scale=jnp.ones(9),
            F_Pos_Bias=jnp.zeros(9),
            F_Pos_Scale=jnp.ones(9),
            F_Up_Plim=jnp.zeros(9) + jnp.inf,
            F_Lwr_Plim=jnp.zeros(9) - jnp.inf,
            F_Rate_Bias=jnp.zeros(9),
            F_Rate_Scale=jnp.ones(9),
            F_Up_Rlim=jnp.zeros(9) + jnp.inf,
            F_Lwr_Rlim=jnp.zeros(9) - jnp.inf,
            F_Accel_Bias=jnp.zeros(9),
            F_Accel_Scale=jnp.ones(9),
            F_Up_Alim=jnp.zeros(9) + jnp.inf,
            F_Lwr_Alim=jnp.zeros(9) - jnp.inf,
            F_Gen_Sig_Sel=jnp.zeros(15),
        )
        # jnp.concatenate([jnp.zeros(9),jnp.zeros(9),jnp.ones(9),jnp.ones(9),jnp.zeros(9),jnp.ones(9),jnp.zeros(9)+jnp.inf,jnp.zeros(9)-jnp.inf,
        #                            jnp.zeros(9),jnp.ones(9),jnp.zeros(9)+jnp.inf,jnp.zeros(9)-jnp.inf,jnp.zeros(9),jnp.ones(9),jnp.zeros(9)+jnp.inf,jnp.zeros(9)-jnp.inf,jnp.zeros(15)])
        return PropAct(EngSpeed=eng_speed, EngAccel=eng_accel, Failure_Engines=eng_failure)
