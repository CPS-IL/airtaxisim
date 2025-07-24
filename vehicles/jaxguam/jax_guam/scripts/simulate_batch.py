import functools as ft

import ipdb
import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import tqdm
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from loguru import logger


def main():
    jax_use_cpu()
    jax_use_double()
    set_logger_format()

    final_time = 45.0

    logger.info("Constructing GUAM...")
    guam = FuncGUAM()
    logger.info("Calling GUAM...")

    batch_size = 4096
    # batch_size = 8192
    # batch_size = 16_384
    state = GuamState.create()
    b_state: GuamState = jtu.tree_map(lambda x: np.broadcast_to(x, (batch_size,) + x.shape).copy(), state)
    T = int(final_time / guam.dt)

    # Perturb the initial state in the x and y directions.
    key0, key1 = jr.split(jr.PRNGKey(0))
    b_state.aircraft[:, 6] = jr.uniform(key0, (batch_size,), minval=-20.0, maxval=20.0)
    b_state.aircraft[:, 7] = jr.uniform(key1, (batch_size,), minval=-20.0, maxval=20.0)

    vmap_step = jax.jit(jax.vmap(ft.partial(guam.step, guam.dt), in_axes=(0, None)))

    def simulate_batch(b_state0) -> GuamState:
        Tb_state = [b_state0]
        b_state = b_state0
        for kk in tqdm.trange(T):
            t = kk * guam.dt
            ref_inputs = lift_cruise_reference_inputs(t)
            b_state = vmap_step(b_state, ref_inputs)
            Tb_state.append(jax2np(b_state))
        bT_state = jtu.tree_map(lambda *args: np.stack(list(args), axis=1), *Tb_state)
        return bT_state

    bT_state = simulate_batch(b_state)

    np.savez("bT_state.npz", aircraft=bT_state.aircraft)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
