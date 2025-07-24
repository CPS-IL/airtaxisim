import ipdb
import jax.numpy as jnp

from jax_guam.classes.SemiWingPropClass import SemiWingPropClass
from jax_guam.classes.WingClass import WingClass
from jax_guam.utils.angle_utils import Ry
from jax_guam.utils.jax_types import Mat31


class WingPropClass:
    def __init__(self, wing: WingClass):
        self.Props = None
        self.c4_h = None
        self.I = None
        self.Wing = wing
        self.mass = self.Wing.mass
        self.b_e = self.Wing.b_e
        self.semi_b_e = self.b_e / 2
        y_diff = (self.Wing.b - self.b_e) / 2
        self.semi_c4_b = jnp.zeros((3, 2))
        # self.semi_c4_b[:,0] = self.Wing.c4_b  - jnp.array([[0], [y_diff], [0]])
        # self.semi_c4_b[:,1] = self.Wing.c4_b  + jnp.array([[0], [y_diff], [0]])
        de = self.Wing.c4_b - jnp.array([[0], [y_diff], [0]])
        su = self.Wing.c4_b + jnp.array([[0], [y_diff], [0]])
        self.semi_c4_b = self.semi_c4_b.T.at[0].set(de.reshape(3)).T  # Note weird
        self.semi_c4_b = self.semi_c4_b.T.at[1].set(su.reshape(3)).T  # Note weird
        self.NP = 0
        self.Left = SemiWingPropClass(
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

        self.Right = SemiWingPropClass(
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

        self.set_h_b(self.Wing.c4_b)
        self.set_tilt_angle(0)
        self.cm_b = self.get_cm_b()
        self.Fx = 0
        self.Fy = 0
        self.Fz = 0
        self.Mx = 0
        self.My = 0
        self.Mz = 0

        self.Fx_x = jnp.zeros(6)
        self.Fy_x = jnp.zeros(6)
        self.Fz_x = jnp.zeros(6)
        self.Mx_x = jnp.zeros(6)
        self.My_x = jnp.zeros(6)
        self.Mz_x = jnp.zeros(6)

        self.Fx_u = jnp.zeros(self.NP + 3)
        self.Fy_u = jnp.zeros(self.NP + 3)
        self.Fz_u = jnp.zeros(self.NP + 3)
        self.Mx_u = jnp.zeros(self.NP + 3)
        self.My_u = jnp.zeros(self.NP + 3)
        self.Mz_u = jnp.zeros(self.NP + 3)

    # def Ry(self, x):
    #     return jnp.array(
    #         [[float(jnp.cos(x)), 0, float(jnp.sin(x))], [0, 1, 0], [float(-jnp.sin(x)), 0, float(jnp.cos(x))]]
    #     )

    def get_w_cm_b(self):
        # if self.h_b is not None:
        #     r = self.Wing.cm_b
        # else:
        # import pdb; pdb.set_trace()
        w_cm_h = self.Wing.cm_b - self.h_b
        r = self.h_b + Ry(self.tilt_angle) @ w_cm_h
        return r

    def get_cm_b(self):
        w_cm_b = self.get_w_cm_b()
        w_m = self.Wing.mass
        r = (1 / self.mass) * w_cm_b * w_m
        return r

    def set_del_a(self, del_a):
        self.del_a = del_a
        self.Left.set_del_a(del_a)
        self.Right.set_del_a(del_a)

    def set_del_f(self, del_f):
        self.del_f = del_f
        self.Left.set_del_f(del_f)
        self.Right.set_del_f(del_f)

    def set_h_b(self, h_b):
        self.c4_h = self.Wing.c4_b - h_b
        self.h_b = h_b
        y_diff = (self.Wing.b - self.b_e) / 2
        self.Left.c4_h = self.c4_h - jnp.array([[0], [y_diff], [0]])
        self.Right.c4_h = self.c4_h - jnp.array([[0], [y_diff], [0]])
        self.Left.h_b = self.h_b
        self.Right.h_b = self.h_b

    def set_tilt_angle(self, tilt_angle):
        assert tilt_angle == 0
        self.tilt_angle = tilt_angle
        self.Left.tilt_angle = tilt_angle
        self.Right.tilt_angle = tilt_angle

    def aero(self, rho: float, uvw: Mat31, om: Mat31, cm_b: Mat31, ders: bool):
        assert isinstance(ders, bool)
        # Get the aerodynamic contributions of the left and right wing and add them up
        self.Left = self.Left.aero(rho, uvw, om, cm_b, ders)
        self.Right = self.Right.aero(rho, uvw, om, cm_b, ders)

        # self.L = self.Left.L + self.Right.L
        # self.D = self.Left.D + self.Right.D
        # self.S = self.Left.S + self.Right.S

        # Calculate the sum of forces and moments
        self.Fx = self.Left.Fx + self.Right.Fx
        self.Fy = self.Left.Fy + self.Right.Fy
        self.Fz = self.Left.Fz + self.Right.Fz
        self.Mx = self.Left.Mx + self.Right.Mx
        self.My = self.Left.My + self.Right.My
        self.Mz = self.Left.Mz + self.Right.Mz
        # import pdb; pdb.set_trace()

        # if ders:
        #     self.Fx_x = self.Left.Fx_x + self.Right.Fx_x
        #     self.Fy_x = self.Left.Fy_x + self.Right.Fy_x
        #     self.Fz_x = self.Left.Fz_x + self.Right.Fz_x
        #     self.Mx_x = self.Left.Mx_x + self.Right.Mx_x
        #     self.My_x = self.Left.My_x + self.Right.My_x
        #     self.Mz_x = self.Left.Mz_x + self.Right.Mz_x

        #     # left side Props, right side props
        #     self.Fx_u = jnp.concatenate([jnp.fliplr(self.Left.Fx_u[:, :self.NP//2]), self.Right.Fx_u[:, :self.NP//2]], axis=1)
        #     # elevator, rudder, tilt
        #     self.Fx_u = jnp.concatenate([self.Fx_u, self.Left.Fx_u[:, self.NP//2:] + self.Right.Fx_u[:, self.NP//2:]], axis=1)

        #     # left side Props, right side props
        #     self.Fy_u = jnp.concatenate([jnp.fliplr(self.Left.Fy_u[:, :self.NP//2]), self.Right.Fy_u[:, :self.NP//2]], axis=1)
        #     # elevator, rudder, tilt
        #     self.Fy_u = jnp.concatenate([self.Fy_u, self.Left.Fy_u[:, self.NP//2:] + self.Right.Fy_u[:, self.NP//2:]], axis=1)

        #     # left side Props, right side props
        #     self.Fz_u = jnp.concatenate([jnp.fliplr(self.Left.Fz_u[:, :self.NP//2]), self.Right.Fz_u[:, :self.NP//2]], axis=1)
        #     # elevator, rudder, tilt
        #     self.Fz_u = jnp.concatenate([self.Fz_u, self.Left.Fz_u[:, self.NP//2:] + self.Right.Fz_u[:, self.NP//2:]], axis=1)

        #     # left side Props, right side props
        #     self.Mx_u = jnp.concatenate([jnp.fliplr(self.Left.Mx_u[:, :self.NP//2]), self.Right.Mx_u[:, :self.NP//2]], axis=1)
        #     # elevator, rudder, tilt
        #     self.Mx_u = jnp.concatenate([self.Mx_u, self.Left.Mx_u[:, self.NP//2:] + self.Right.Mx_u[:, self.NP//2:]], axis=1)

        #     # left side Props, right side props
        #     self.My_u = jnp.concatenate([jnp.fliplr(self.Left.My_u[:, :self.NP//2]), self.Right.My_u[:, :self.NP//2]], axis=1)
        #     # elevator, rudder, tilt
        #     self.My_u = jnp.concatenate([self.My_u, self.Left.My_u[:, self.NP//2:] + self.Right.My_u[:, self.NP//2:]], axis=1)

        #     # left side Props, right side props
        #     self.Mz_u = jnp.concatenate([jnp.fliplr(self.Left.Mz_u[:, :self.NP//2]), self.Right.Mz_u[:, :self.NP//2]], axis=1)
        #     # elevator, rudder, tilt
        #     self.Mz_u = jnp.concatenate([self.Mz_u, self.Left.Mz_u[:, self.NP//2:] + self.Right.Mz_u[:, self.NP//2:]], axis=1)
        return self
