import jax.numpy as jnp

from jax_guam.functional.semi_wing_prop import FuncSemiWingProp
from jax_guam.utils.jax_types import FloatScalar, FMVec, IntScalar, Vec3_1


class FuncVerticalTail:
    def __init__(self, airfoil, coeff, span, c, y_rudder, c4_b, mass, I, cm_b):
        # airfoil = airfoil
        coeff = coeff
        b = span
        c = c
        y_rudder = y_rudder
        self.c4_b = c4_b

        self.tail = FuncSemiWingProp(
            "Right", None, coeff, b, c, jnp.pi / 2, y_rudder, [0, 0], self.c4_b.reshape((3, 1))
        )

        self.mass = mass
        self.I = I
        self.cm_b = cm_b

    def aero(
        self,
        rho: float,
        uvw: Vec3_1,
        om: Vec3_1,
        cm_b: Vec3_1,
        ders: bool,
        del_a: FloatScalar,
        del_f: FloatScalar,
        tilt_angle: FloatScalar | IntScalar,
    ) -> FMVec:
        assert isinstance(ders, bool) and ders is False
        return self.tail.aero(rho, uvw, om, cm_b, ders, del_a, del_f, tilt_angle)
