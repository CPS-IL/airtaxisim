import ipdb
from loguru import logger
import jax.numpy as jnp

from jax_guam.classes.SemiWingPropClass import SemiWingPropClass
from jax_guam.classes.VerticalTailClass import VerticalTailClass
from jax_guam.classes.WingClass import WingClass


class TailClass:
    def __init__(self, hTail: WingClass, vTail: VerticalTailClass):
        self.tilt_angle = 0 # Note: set to const zero.
        self.del_e = 0 # Note: is set.
        self.del_r = 0 # Note: is set.
        # look for setters
        self.om_prop = None
        # self.Thrust = None # Note: UNUSED
        self.h_b = None # Is never set!
        # self.Props = None # Note: Unused
        self.semi_b_e = None
        # self.c4_h = None # Note: Unused
        # self.I = None # Note: Unused

        self.Horz = hTail
        self.b_e = self.Horz.b_e
        self.semi_b_e = self.b_e / 2
        y_diff = (self.Horz.b - self.b_e) / 2
        self.semi_c4_b = jnp.zeros((3, 2))
        # self.semi_c4_b[:,0] = self.Horz.c4_b  - jnp.array([[0], [y_diff], [0]])
        # self.semi_c4_b[:,1] = self.Horz.c4_b  + jnp.array([[0], [y_diff], [0]])
        de = self.Horz.c4_b - jnp.array([[0], [y_diff], [0]])
        su = self.Horz.c4_b + jnp.array([[0], [y_diff], [0]])
        self.semi_c4_b = self.semi_c4_b.T.at[0].set(de.reshape(3)).T  # Note weird
        self.semi_c4_b = self.semi_c4_b.T.at[1].set(su.reshape(3)).T  # Note weird
        self.NP = 0
        self.Left = SemiWingPropClass(
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

        self.Right = SemiWingPropClass(
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

    def get_ht_cm_b(self):
        if self.h_b is None:
            r = self.Horz.cm_b
        else:
            ht_cm_h = self.Horz.cm_b - self.h_b
            r = self.h_b + self.Ry(self.tilt_angle) * ht_cm_h
        return r

    def get_cm_b(self):
        ht_cm_b = self.get_ht_cm_b()
        ht_m = self.Horz.mass

        vt_cm_b = self.Vert.cm_b
        vt_m = self.Vert.mass

        r = (1 / self.mass) * sum([ht_cm_b * ht_m, vt_cm_b * vt_m])
        return r

    # Note: Not used!
    # def Rx(self, x):
    #     return jnp.array([[0, 0], [0, jnp.cos(x), -jnp.sin(x)], [0, jnp.sin(x), jnp.cos(x)]])
    #
    # def Ry(self, x):
    #     return jnp.array(
    #         [[float(jnp.cos(x)), 0, float(jnp.sin(x))], [0, 1, 0], [float(-jnp.sin(x)), 0, float(jnp.cos(x))]]
    #     )
    #
    # def Rz(self, x):
    #     return jnp.array([[jnp.cos(x), -jnp.sin(x), 0], [jnp.sin(x), jnp.cos(x), 0], [0, 0, 1]])
    #
    # def hat(self, x):
    #     return jnp.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    def set_del_e(self, del_e):
        self.del_e = del_e
        self.Left.set_del_f(del_e)
        self.Right.set_del_f(del_e)

    def set_del_r(self, del_r):
        self.del_r = del_r
        self.Vert.tail.set_del_f(del_r)

    def set_tilt_angle(self, tilt_angle: float):
        self.tilt_angle = tilt_angle
        self.Left.tilt_angle = tilt_angle
        self.Right.tilt_angle = tilt_angle

    def aero(self, rho, uvw, om, cm_b, ders: bool):
        assert isinstance(ders, bool) and ders is False
        # logger.info("uvw.shape: {}".format(uvw.shape))
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

        # Get aerodynamics of each component
        self.Left = self.Left.aero(rho, v_l, om, cm_b, ders)
        self.Right = self.Right.aero(rho, v_r, om, cm_b, ders)
        self.Vert = self.Vert.aero(rho, v_b, om, cm_b, ders)

        # Sum the forces and moments
        self.Fx = self.Left.Fx + self.Right.Fx + self.Vert.Fx
        self.Fy = self.Left.Fy + self.Right.Fy + self.Vert.Fy
        self.Fz = self.Left.Fz + self.Right.Fz + self.Vert.Fz
        self.Mx = self.Left.Mx + self.Right.Mx + self.Vert.Mx
        self.My = self.Left.My + self.Right.My + self.Vert.My
        self.Mz = self.Left.Mz + self.Right.Mz + self.Vert.Mz

        # if ders:
        #     self.Fx_x = self.Left.Fx_x + self.Right.Fx_x + self.Vert.Fx_x
        #     self.Fy_x = self.Left.Fy_x + self.Right.Fy_x + self.Vert.Fy_x
        #     self.Fz_x = self.Left.Fz_x + self.Right.Fz_x + self.Vert.Fz_x
        #     self.Mx_x = self.Left.Mx_x + self.Right.Mx_x + self.Vert.Mx_x
        #     self.My_x = self.Left.My_x + self.Right.My_x + self.Vert.My_x
        #     self.Mz_x = self.Left.Mz_x + self.Right.Mz_x + self.Vert.Mz_x

        #     # Left side Props, right side props
        #     self.Fx_u =  jnp.concatenate([np.flipud(self.Left.Fx_u[:self.NP // 2]), self.Right.Fx_u[:self.NP // 2]])
        #     # Elevator, rudder, tilt
        #     self.Fx_u =  jnp.concatenate([self.Fx_u, self.Left.Fx_u[:, self.NP // 2] + self.Right.Fx_u[self.NP // 2], self.Vert.Fx_u[0], self.Left.Fx_u[-1] + self.Right.Fx_u[-1]])

        #     # Left side Props, right side props
        #     self.Fy_u =  jnp.concatenate([np.flipud(self.Left.Fy_u[:self.NP // 2]), self.Right.Fy_u[:self.NP // 2]])
        #     # Elevator, rudder, tilt
        #     self.Fy_u =  jnp.concatenate([self.Fy_u, self.Left.Fy_u[:, self.NP // 2] + self.Right.Fy_u[self.NP // 2], self.Vert.Fy_u[0], self.Left.Fy_u[-1] + self.Right.Fy_u[-1]])

        #     # Left side Props, right side props
        #     self.Fz_u =  jnp.concatenate([np.flipud(self.Left.Fz_u[:self.NP // 2]), self.Right.Fz_u[:self.NP // 2]])
        #     # Elevator, rudder, tilt
        #     self.Fz_u =  jnp.concatenate([self.Fz_u, self.Left.Fz_u[:, self.NP // 2] + self.Right.Fz_u[self.NP // 2], self.Vert.Fz_u[0], self.Left.Fz_u[-1] + self.Right.Fz_u[-1]])

        #     # Left side Props, right side props
        #     self.Mx_u =  jnp.concatenate([np.flipud(self.Left.Mx_u[:self.NP // 2]), self.Right.Mx_u[:self.NP // 2]])
        #     # Elevator, rudder, tilt
        #     self.Mx_u =  jnp.concatenate([self.Mx_u, self.Left.Mx_u[:, self.NP // 2] + self.Right.Mx_u[self.NP // 2], self.Vert.Mx_u[0], self.Left.Mx_u[-1] + self.Right.Mx_u[-1]])

        #     # Left side Props, right side props
        #     self.My_u =  jnp.concatenate([np.flipud(self.Left.My_u[:self.NP // 2]), self.Right.My_u[:self.NP // 2]])
        #     # Elevator, rudder, tilt
        #     self.My_u =  jnp.concatenate([self.My_u, self.Left.My_u[:, self.NP // 2] + self.Right.My_u[self.NP // 2], self.Vert.My_u[0], self.Left.My_u[-1] + self.Right.My_u[-1]])

        #     # Left side Props, right side props
        #     self.Mz_u =  jnp.concatenate([np.flipud(self.Left.Mz_u[:self.NP // 2]), self.Right.Mz_u[:self.NP // 2]])
        #     # Elevator, rudder, tilt
        #     self.Mz_u =  jnp.concatenate([self.Mz_u, self.Left.Mz_u[:, self.NP // 2] + self.Right.Mz_u[self.NP // 2], self.Vert.Mz_u[0], self.Left.Mz_u[-1] + self.Right.Mz_u[-1]])

        return self
