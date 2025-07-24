from typing import Sequence

from jax_guam.utils.jax_types import Af2, Mat1, Mat3, Mat31, Vec2, Vec3_1


class WingClass:
    """Pure data class"""

    def __init__(
        self,
        airfoil: Af2,
        coeff: Mat1,
        b: Sequence[float],
        c: Sequence[float],
        gamma: float,
        y_flap: Vec2,
        y_aileron: Vec2,
        c4_b: Vec3_1,
        mass: float,
        I: Mat3,
        cm_b: Vec3_1,
    ):
        self.airfoil: Af2 = airfoil
        self.coeff: Mat1 = coeff
        self.b: float = b[0]
        self.b_e: float = b[1]
        self.c_root: float = c[0]
        self.c_tip: float = c[1]
        # b = b
        # if len(b) == 1:
        #   self.b = b
        #   self.b_e = b
        # elif len(b) == 2:
        #   self.b = b[0]
        #   self.b_e = b[1]
        # else
        #   error('Too many arguments in wing span length')

        # c = c
        # if len(c) == 1
        #   self.c_root  = c
        #   self.c_tip   = c
        # elif len(c) == 2
        #   self.c_root = c[0]
        #   self.c_tip = c[1]
        # else
        #   error('Too many arguments in chord length')

        self.gamma: float = gamma
        self.y_flap: Vec2 = y_flap
        self.y_aileron: Vec2 = y_aileron
        self.c4_b: Vec3_1 = c4_b
        self.mass: float = mass
        self.I: Mat3 = I
        self.cm_b: Vec3_1 = cm_b
