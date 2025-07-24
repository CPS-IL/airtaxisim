import inspect

import ipdb
import jax.numpy as jnp
import numpy as np
from loguru import logger


def log_local_shapes():
    import inspect

    frame = inspect.currentframe()
    try:
        caller_fn_name = frame.f_back.f_code.co_name
        for k, v in frame.f_back.f_locals.items():
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                logger.info("[{}] {}: {}".format(caller_fn_name, k, v.shape))
            else:
                logger.info("[{}] {}: {}".format(caller_fn_name, k, type(v)))
        ipdb.set_trace()
    finally:
        del frame
