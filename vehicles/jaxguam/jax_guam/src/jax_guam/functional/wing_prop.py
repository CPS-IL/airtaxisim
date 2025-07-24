import jax.numpy as jnp
import numpy as np

from jax_guam.classes.WingClass import WingClass
from jax_guam.functional.semi_wing_prop import FuncSemiWingProp
from jax_guam.utils.angle_utils import Ry
from jax_guam.utils.jax_types import FloatScalar, FMVec, IntScalar, Mat31, Vec3_1


class FuncWingProp:
    def __init__(self, wing: WingClass):
        self.Wing = wing
        self.mass = self.Wing.mass
        self.b_e = self.Wing.b_e
        self.semi_b_e = self.b_e / 2
        y_diff = (self.Wing.b - self.b_e) / 2
        # self.semi_c4_b[:,0] = self.Wing.c4_b  - jnp.array([[0], [y_diff], [0]])
        # self.semi_c4_b[:,1] = self.Wing.c4_b  + jnp.array([[0], [y_diff], [0]])
        de = self.Wing.c4_b - np.array([[0], [y_diff], [0]])
        su = self.Wing.c4_b + np.array([[0], [y_diff], [0]])
        assert de.shape == su.shape == (3, 1)

        # Note: Old
        # self.semi_c4_b = jnp.zeros((3, 2))
        # self.semi_c4_b = self.semi_c4_b.T.at[0].set(de.reshape(3)).T  # Note weird
        # self.semi_c4_b = self.semi_c4_b.T.at[1].set(su.reshape(3)).T  # Note weird
        self.semi_c4_b = np.concatenate([de, su], axis=1)
        assert self.semi_c4_b.shape == (3, 2)

        self.NP = 0
        self.Left = FuncSemiWingProp(
            "Left",
            self.Wing.airfoil,
            self.Wing.coeff,
            self.semi_b_e,
            [self.Wing.c_root, self.Wing.c_tip],
            self.Wing.gamma,
            self.Wing.y_flap,
            self.Wing.y_aileron,
            self.semi_c4_b[:, 0].reshape((3, 1)),
        )

        self.Right = FuncSemiWingProp(
            "Right",
            self.Wing.airfoil,
            self.Wing.coeff,
            self.semi_b_e,
            [self.Wing.c_root, self.Wing.c_tip],
            self.Wing.gamma,
            self.Wing.y_flap,
            self.Wing.y_aileron,
            self.semi_c4_b[:, 1].reshape((3, 1)),
        )

        # Extracted from set_h_b.
        # Note: This modifies c4_h and h_b of Left and Right.
        self.c4_h, self.h_b = self.set_h_b(self.Wing.c4_b)

        self.cm_b: Vec3_1 = self.get_cm_b(tilt_angle=0)

    def set_h_b(self, h_b: Vec3_1) -> tuple[Vec3_1, Vec3_1]:
        self_c4_h = self.Wing.c4_b - h_b
        self_h_b = h_b
        y_diff = (self.Wing.b - self.b_e) / 2
        self.Left.c4_h = self_c4_h - jnp.array([[0], [y_diff], [0]])
        self.Right.c4_h = self_c4_h - jnp.array([[0], [y_diff], [0]])
        self.Left.h_b = self_h_b
        self.Right.h_b = self_h_b

        return self_c4_h, self_h_b

    def get_w_cm_b(self, tilt_angle: FloatScalar | IntScalar) -> Vec3_1:
        # if self.h_b is not None:
        #     r = self.Wing.cm_b
        # else:
        # import pdb; pdb.set_trace()
        w_cm_h = self.Wing.cm_b - self.h_b
        r = self.h_b + Ry(tilt_angle) @ w_cm_h
        return r

    def get_cm_b(self, tilt_angle: FloatScalar | IntScalar) -> Vec3_1:
        w_cm_b = self.get_w_cm_b(tilt_angle)
        w_m = self.Wing.mass
        r = (1 / self.mass) * w_cm_b * w_m
        return r

    def aero(
        self,
        rho: float,
        uvw: Vec3_1,
        om: Vec3_1,
        cm_b: Vec3_1,
        ders: bool,
        del_a: FloatScalar,
        del_f: FloatScalar,
        tilt_angle: FloatScalar,
    ) -> FMVec:
        assert isinstance(ders, bool) and ders is False

        fm_left = self.Left.aero(rho, uvw, om, cm_b, ders, del_a, del_f, tilt_angle)
        fm_right = self.Right.aero(rho, uvw, om, cm_b, ders, del_a, del_f, tilt_angle)
        return fm_left + fm_right
