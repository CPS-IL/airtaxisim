import jax.numpy as jnp
from loguru import logger

from jax_guam.utils.jax_types import Mat3, Vec3_1


class MassClass:
    def __init__(self, mass: float, I: Mat3, cm_b: Vec3_1):
        assert isinstance(mass, float) and I.shape == (3, 3) and cm_b.shape == (3, 1)
        # assert isinstance(mass, float) and I.shape == (3, 3)
        self.mass = mass
        self.I = I
        self.cm_b = cm_b
