import jax.numpy as jnp
import numpy as np

from jax_guam.classes.WingClass import WingClass
from jax_guam.functional.semi_wing_prop import FuncSemiWingProp
from jax_guam.functional.vertical_tail import FuncVerticalTail
from jax_guam.utils.jax_types import FloatScalar, FMVec, Vec3_1


class FuncTail:
    def __init__(self, hTail: WingClass, vTail: FuncVerticalTail):
        self.h_b = None

        self.Horz = hTail
        self.b_e = self.Horz.b_e
        self.semi_b_e = self.b_e / 2
        y_diff = (self.Horz.b - self.b_e) / 2
        de = self.Horz.c4_b - jnp.array([[0], [y_diff], [0]])
        su = self.Horz.c4_b + jnp.array([[0], [y_diff], [0]])
        assert de.shape == su.shape == (3, 1)

        # self.semi_c4_b = self.semi_c4_b.T.at[0].set(de.reshape(3)).T  # Note weird
        # self.semi_c4_b = self.semi_c4_b.T.at[1].set(su.reshape(3)).T  # Note weird
        self.semi_c4_b = np.concatenate([de, su], axis=1)

        self.NP = 0
        self.Left = FuncSemiWingProp(
            "Left",
            self.Horz.airfoil,
            self.Horz.coeff,
            self.semi_b_e,
            [self.Horz.c_root, self.Horz.c_tip],
            self.Horz.gamma,
            self.Horz.y_flap,
            self.Horz.y_aileron,
            self.semi_c4_b[:, 0].reshape((3, 1)),
        )
        self.Right = FuncSemiWingProp(
            "Right",
            self.Horz.airfoil,
            self.Horz.coeff,
            self.semi_b_e,
            [self.Horz.c_root, self.Horz.c_tip],
            self.Horz.gamma,
            self.Horz.y_flap,
            self.Horz.y_aileron,
            self.semi_c4_b[:, 1].reshape((3, 1)),
        )
        self.Vert = vTail
        self.mass = self.Horz.mass + self.Vert.mass
        self.cm_b: Vec3_1 = self.get_cm_b()

    def get_ht_cm_b(self):
        assert self.h_b is None
        if self.h_b is None:
            r = self.Horz.cm_b
        else:
            raise RuntimeError("")
            # ht_cm_h = self.Horz.cm_b - self.h_b
            # r = self.h_b + self.Ry(self.tilt_angle) * ht_cm_h

        return r

    def get_cm_b(self) -> Vec3_1:
        ht_cm_b = self.get_ht_cm_b()
        ht_m = self.Horz.mass

        vt_cm_b = self.Vert.cm_b
        vt_m = self.Vert.mass

        r = (1 / self.mass) * sum([ht_cm_b * ht_m, vt_cm_b * vt_m])
        return r

    def aero(
        self,
        rho: float,
        uvw: Vec3_1,
        om: Vec3_1,
        cm_b: Vec3_1,
        ders: bool,
        del_e: FloatScalar,
        del_r: FloatScalar,
        tilt_angle: FloatScalar,
    ) -> FMVec:
        assert isinstance(ders, bool) and ders is False
        assert uvw.shape == (3, 1)
        if len(uvw) == 3:
            v_b = uvw
            v_l = uvw
            v_r = uvw
        elif len(uvw) > 3:
            v_b = uvw[:, 0]
            v_l = uvw[:, 1]
            v_r = uvw[:, 2]
        else:
            raise ValueError("The input v is the wrong size")

        # del_e -> Left.del_f, Right.del_f
        lr_del_f = del_e

        # del_r -> Vert.tail.del_f
        tail_del_f = del_r

        # tilt_angle -> Left.tilt_angle, Right.tilt_angle
        lr_tilt_angle = tilt_angle

        # Left.del_a, Tail.del_a and Tail.tilt_angle are not set, so they are zero.
        lr_del_a, tail_del_a, tail_tilt_angle = 0, 0, 0

        # Get aerodynamics of each component
        fm_left = self.Left.aero(rho, v_l, om, cm_b, ders, lr_del_a, lr_del_f, lr_tilt_angle)
        fm_right = self.Right.aero(rho, v_r, om, cm_b, ders, lr_del_a, lr_del_f, lr_tilt_angle)
        fm_vert = self.Vert.aero(rho, v_b, om, cm_b, ders, tail_del_a, tail_del_f, tail_tilt_angle)

        # Sum the forces and moments
        FM = fm_left + fm_right + fm_vert
        assert FM.shape == (6,)
        return FM
