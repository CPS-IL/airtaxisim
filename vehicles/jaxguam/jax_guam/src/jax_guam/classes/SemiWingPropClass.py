import pdb

import ipdb
import numpy as np
from typing import Literal

import jax.numpy as jnp
from loguru import logger

from jax_guam.classes.base_functions.FM_body import FM_body
from jax_guam.utils.angle_utils import Rx, Ry
from jax_guam.utils.debug import log_local_shapes
from jax_guam.utils.jax_types import Mat1, Vec1, Vec2, Mat31, Af1, Af3, Af8, Af4, Af6, Vec3_1, FloatScalar


class SemiWingPropClass:
    Thrust: Vec1

    def __init__(
        self,
        lr: Literal["Left", "Right"],
        airfoil,
        coeff: Mat1,
        span: float,
        c: list[float],
        angle: float,
        y_flap: Vec2,
        y_aileron: Vec2,
        c4_b: Mat31,
    ):
        self.strip_width = 0.2
        self.del_f = 0  # flap deflection angle
        self.del_a = 0  # aileron deflection angle
        self.tau = 0.6  # flap/aileron effectiveness
        self.NS = jnp.floor(span / self.strip_width)
        self.NS = int(self.NS)

        logger.info("NS: {}".format(self.NS))

        self.yc: Af1 = np.zeros((self.NS, 1))
        self.Si: Af1 = np.zeros((self.NS, 1))
        self.Aki = np.zeros((self.NS, 1))
        # Note: This is zero!
        self.Ali = np.zeros((self.NS, 1))
        self.k_idx: Af1 = np.zeros((self.NS, 1))
        # Note: This is zero!
        self.l_idx = np.zeros((self.NS, 1))
        # Note: This is zero!
        self.Tki = np.zeros((self.NS, 1))
        # Note: This is zero!
        self.Tli = np.zeros((self.NS, 1))
        self.flap_idx: Af1 = np.zeros((self.NS, 1))
        self.aileron_idx = np.zeros((self.NS, 1))
        self.deli = jnp.zeros((self.NS, 1))
        # Note: This is zero!
        self.Li = np.zeros((self.NS, 1))
        # Note: This is zero!
        self.Di = np.zeros((self.NS, 1))
        # Note: This is zero!
        self.alfi = np.zeros((self.NS, 1))

        self.hasProp = False

        # check left or right side
        if lr == "Left":
            self.sgn = -1
        else:
            self.sgn = 1

        # import pdb pdb.set_trace()
        # assign wing parameters
        # self.airfoil = airfoil # Note: Unused!
        self.aero_coefs_pp: Mat1 = coeff
        self.b = span
        chord = c
        self.gamma = angle
        self.y_flap = y_flap
        self.y_aileron = y_aileron
        self.h_b: Mat31 = c4_b

        # if no propellers are specified then define one prop that is
        # centered on the wing and who's diamater is equal to the span.
        # Setting the thrust equal to zero will cause the "ghost" propeller
        # to have no effect on the aerodynamics.
        self.prop_coef = np.zeros((3, 2))
        self.spin = 1
        self.D_prop = self.b
        self.C_prop = self.b / 2
        self.NP = 0
        self.Ct_prop = np.zeros((3, 1))
        self.Cq_prop = np.zeros((3, 1))

        # assign the chord length
        if len(chord) == 1:
            self.c_root = chord
            self.c_tip = chord
        elif len(chord) == 2:
            self.c_root = chord[0]
            self.c_tip = chord[1]
        else:
            raise Exception("Too many arguments in chord length")

        # assign the quarter chord position in the body frame. When
        # setting up the wing we assume that if the wing rotates, that
        # it is hinged at the quarter chord and the initial tilt angle
        # is zero.
        self.c4_h: Vec3_1 = jnp.zeros((3, 1))
        self.tilt_angle: int | float = 0

        # Finally set up the strips
        self.SetupStrips()
        #        if self.NP  > 0
        assert not self.hasProp
        if self.hasProp:
            self.set_thrust(jnp.zeros(self.NP))  # initialize thrust at zero
            self.Torque = jnp.zeros(self.NP)
            self.om_prop = jnp.zeros((self.NP, 1))
            self.e_b = jnp.zeros((3, self.NP))
        else:
            self.set_thrust(np.zeros(1))
            self.Torque = np.zeros(1)
            self.om_prop = np.zeros(1)
            self.e_b = np.zeros((3, 1))

        # # Note: self.A has already been set to the same value in SetupStrips
        # self.A = jnp.pi * self.D_prop**2 / 4  # propeller disk areas
        self.c4_b: Vec3_1 = self.get_c4_b()

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

        self.Fx_u = jnp.zeros(self.NP + 3)  # NP + 3 surface inputs
        self.Fy_u = jnp.zeros(self.NP + 3)
        self.Fz_u = jnp.zeros(self.NP + 3)
        self.Mx_u = jnp.zeros(self.NP + 3)
        self.My_u = jnp.zeros(self.NP + 3)
        self.Mz_u = jnp.zeros(self.NP + 3)

    def SetupStrips(self):
        # Only called during init.

        NS = self.NS

        # Make local copies of everything
        c_root = self.c_root
        c_tip = self.c_tip
        b = self.b
        D_prop = self.D_prop
        C_prop = self.C_prop

        # Define propeller properties
        # Find the length of the overlapped regions
        # l = (C_prop[:-1] + D_prop[:-1] / 2) - (C_prop[1:] - D_prop[1:] / 2) #

        Np = 1  # Number of props per wing
        NA = Np * 2 - 1  # Number of Areas (includes overlap areas)

        A = jnp.pi * D_prop**2 / 4  # Propeller disk areas
        self.A = A

        ymax = max([b, C_prop + D_prop / 2])

        dy = ymax / NS  # Width of each strip

        y = np.arange(0, ymax + dy, dy)  # y points across the wing and props
        yw = y[y <= b]  # y point of just the wing

        cw = c_root - yw / b * (c_root - c_tip)  # Chord length at each point
        Sw = dy * (cw[:-1] + cw[1:]) / 2  # Area of each strip

        cw = cw.reshape((cw.shape[0], 1))
        Sw = Sw.reshape((Sw.shape[0], 1))
        if y[y > b].shape[0] == 0:
            ci = cw
            Si = Sw
        else:
            ci = np.concatenate([cw, np.zeros((y[y > b].shape[0], 1))], axis=0)
            Si = np.concatenate([Sw, np.zeros((y[y > b].shape[0], 1))], axis=0)

        # R1 = D_prop[:-1] / 2 #
        # R2 = D_prop[1:] / 2 #

        # d = R1 + R2 - l #
        # d1 = (d**2 + R1**2 - R2**2) / (2 * d)  #
        # d2 = d - d1  #

        # a = jnp.real(jnp.sqrt(R1**2 - d1**2))  #
        # th1 = jnp.real(2 * jnp.arccos(d1 / R1))  #
        # th2 = jnp.real(2 * jnp.arccos(d2 / R2))  #
        # A1 = 0.5 * R1**2 * th1 - a * d1  #
        # A2 = 0.5 * R2**2 * th2 - a * d2  #

        # Al = A1 + A2  #
        # AL = jnp.concatenate([jnp.array([0]), Al, jnp.array([0])], axis=0)
        # L = jnp.concatenate([jnp.array([0]), l, jnp.array([0])], axis=0)
        AL = np.array([0, 0])
        L = np.array([0, 0])

        Ak = A - AL[:-1] - AL[1:]
        # odd = jnp.arange(1, NA + 1, step=2)
        # even = jnp.arange(2, NA + 1, step=2)
        odd = 0
        # dY = jnp.zeros(NA)
        # dY[odd] = D_prop - float(L[:-1] * (L[:-1] > 0) - L[1:] * (L[1:] > 0))
        # dY = dY.at[odd].set(D_prop - float(L[:-1] * (L[:-1] > 0) - L[1:] * (L[1:] > 0))) #Note weird bug
        dY = jnp.zeros((NA, 1))
        dY = dY.at[odd].set(D_prop - float(L[:-1] * (L[:-1] > 0) - L[1:] * (L[1:] > 0)))[0]
        # dY[even] = jnp.abs(l)
        Y = float(jnp.cumsum(dY)) + (C_prop - D_prop / 2)

        Di = Y

        zt = jnp.zeros_like(y)
        k_idx = jnp.zeros_like(y)
        for i in range(Np, 0, -1):
            # zt[idx] = jnp.sqrt(D_prop**2 / 4 - (y[idx] - C_prop)**2)
            # k_idx[idx] = i
            zt = zt.at[y < Di].set(jnp.sqrt(D_prop**2 / 4 - (y[y < Di] - C_prop) ** 2))
            k_idx = k_idx.at[y < Di].set(i)
        k_idx = k_idx * (zt > 0)
        k_idx = k_idx[:-1]

        zl = jnp.zeros_like(y)
        l_idx = jnp.zeros_like(y)
        # for i in range(Np - 1):
        #     j = (y < Di[i]) & (y > (Di[i] - d1[i])) -1
        #     k = (y > Di[i]) & (y < (Di[i] + d2[i])) -1
        #     j = j[j>=0]
        #     k = k[k>=0]

        #     zl[j] = jnp.sqrt(D_prop[i + 1]**2 / 4 - (y[j] - C_prop[i + 1])**2)
        #     zl[k] = jnp.sqrt(D_prop[i]**2 / 4 - (y[k] - C_prop[i])**2)

        #     l_idx[j] = i
        #     l_idx[k] = i
        l_idx = l_idx * (zl > 0)
        l_idx = l_idx[:-1]

        Ali = (zl[:-1] + zl[1:]) * dy
        zp = zt - zl
        Aki = (zp[:-1] + zp[1:]) * dy

        self.yc = y[1:] - np.diff(y) / 2  # y distance from root to center of strip
        self.cc = (ci[:-1] + ci[1:]) / 2  # wing strip center cord lengths

        # self.A = A   # Disk area of each prop
        self.Ak = Ak  # propeller non-overlapped areas
        # self.Al = Al # propeller overlapped areas

        self.Si = Si  # wing strip areas
        self.Aki = np.expand_dims(Aki, axis=1)  # propeller non overlapped strip areas
        self.Ali = np.expand_dims(Ali, axis=1)  # propeller over lapped strip areas

        self.k_idx = np.expand_dims(k_idx, axis=1)
        self.l_idx = np.expand_dims(l_idx, axis=1)

        self.Tki = np.zeros_like(k_idx)
        self.Tli = np.zeros_like(l_idx)

        ## Save the flap and aileron indices
        self.flap_idx = (self.yc >= self.y_flap[0]) & (self.yc <= self.y_flap[1])
        self.aileron_idx = (self.yc >= self.y_aileron[0]) & (self.yc <= self.y_aileron[1])
        index = ~(self.flap_idx | self.flap_idx)

        # mark wing strips with out defelcting surfaces
        # self.deli[index] = 255
        self.deli = self.deli.at[index].set(255)

    def get_c4_b(self) -> Vec3_1:
        assert self.tilt_angle == 0
        r = self.h_b + Ry(self.tilt_angle) @ self.c4_h
        return r

    def c4_c(self, cm_b: Mat31) -> Mat31:
        return self.c4_b - cm_b

    # def Rx(self, x):
    #     return jnp.array(
    #         [[1, 0, 0], [0, float(jnp.cos(x)), -float(jnp.sin(x))], [0, float(jnp.sin(x)), float(jnp.cos(x))]]
    #     )
    #
    # def Ry(self, x):
    #     return jnp.array(
    #         [[float(jnp.cos(x)), 0, float(jnp.sin(x))], [0, 1, 0], [float(-jnp.sin(x)), 0, float(jnp.cos(x))]]
    #     )

    # def Rz(self, x):
    #     return jnp.array([[jnp.cos(x), -jnp.sin(x), 0],
    #                     [jnp.sin(x), jnp.cos(x), 0],
    #                     [0, 0, 1]])

    def hat(self, x):
        return jnp.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    def set_del_a(self, del_a: FloatScalar):
        flap = self.flap_idx * self.aileron_idx * self.del_f
        # self.deli[self.aileron_idx] = flap[self.aileron_idx] - self.sgn * del_a
        self.deli = self.deli.at[self.aileron_idx].set(jnp.expand_dims(flap[self.aileron_idx] - self.sgn * del_a, 1))
        self.del_a = del_a

    def set_del_f(self, del_f: FloatScalar):
        aileron = self.flap_idx * self.aileron_idx * self.del_a
        # self.deli[self.flap_idx] = del_f - self.sgn * aileron(self.flap_idx)
        self.deli = self.deli.at[self.flap_idx].set(jnp.expand_dims(del_f - self.sgn * aileron[self.flap_idx], 1))
        self.del_f = del_f

    def set_thrust(self, thrust: Vec1):
        # CALLED ONCE
        assert np.all(thrust == np.zeros(1))

        self.Thrust = thrust
        idx = self.k_idx > 0
        T = np.ones(self.k_idx[idx].shape) * self.Thrust
        A = np.ones(self.k_idx[idx].shape) * self.A
        Aki = self.Aki[idx]
        # self.Tki[idx] = T*Aki/A
        # self.Tki = self.Tki.at[idx].set(jnp.expand_dims(T*Aki/A,1))

        # self.Tki = self.Tki.at[(idx == 1).reshape(idx.shape[0])].set(T * Aki / A)
        self.Tki[(idx == 1).reshape(idx.shape[0])] = T * Aki / A
        assert np.all(self.Tki == 0.0)

        # idx = self.l_idx >0
        # Tl = self.OverlapThrust()
        # Tl = Tl[self.l_idx[idx]]
        # Al = self.Al[self.l_idx[idx]]
        # Ali = self.Ali[idx]
        # # self.Tli[idx] = Tl*Ali/Al
        # self.Tli = self.Tli.at[idx==1].set( Tl*Ali/Al)

    # # UNUSED
    # def OverlapThrust(self):
    #     Tbar = (self.Thrust[:-1] + self.Thrust[1:]) * (
    #         (self.A[:-1] + self.A[1:] - self.Al) / (self.A[:-1] + self.A[1:])
    #     ) ** (1 / 3)
    #     Tl = (
    #         Tbar
    #         - self.Thrust[:-1] * (self.A[:-1] - self.Al) / self.A[:-1]
    #         - self.Thrust[1:] * (self.A[1:] - self.Al) / self.A[1:]
    #     )
    #     return Tl

    def aero(self, rho: float, uvw: Mat31, om: Mat31, cm_b: Mat31, ders: bool):
        assert isinstance(ders, bool) and ders is False
        c4_c: Mat31 = self.c4_c(cm_b)
        b: Af3 = self.sgn * (Rx(-self.sgn * self.gamma) @ jnp.array([0 * self.yc, self.yc, 0 * self.yc])).T + jnp.tile(
            c4_c.T, (self.NS, 1)
        )
        w: Af8 = jnp.column_stack(
            (
                b,
                self.Aki,
                self.Ali,
                self.Si,
                jnp.tile(rho, (self.NS, 1)),
                jnp.tile(-self.sgn * self.gamma, (self.NS, 1)),
            )
        )
        #
        # logger.info("yc: {}".format(np.all(self.yc == 0.0)))
        # logger.info("Si: {}".format(np.all(self.Si == 0.0)))
        # logger.info("Aki: {}".format(np.all(self.Aki == 0.0)))
        # logger.info("Ali: {}".format(np.all(self.Ali == 0.0)))
        # logger.info("k_idx: {}".format(np.all(self.k_idx == 0.0)))
        # logger.info("l_idx: {}".format(np.all(self.l_idx == 0.0)))
        # logger.info("Tki: {}".format(np.all(self.Tki == 0.0)))
        # logger.info("Tli: {}".format(np.all(self.Tli == 0.0)))
        # logger.info("flap_idx: {}".format(np.all(self.flap_idx == 0.0)))
        # logger.info("aileron_idx: {}".format(np.all(self.aileron_idx == 0.0)))
        # logger.info("deli: {}".format(np.all(self.deli == 0.0)))
        # logger.info("Li: {}".format(np.all(self.Li == 0.0)))
        # logger.info("Di: {}".format(np.all(self.Di == 0.0)))
        # logger.info("alfi: {}".format(np.all(self.alfi == 0.0)))

        # if self.hasProp:
        # b_prop = self.sgn * (jnp.array([0 * self.C_prop, self.C_prop, 0 * self.C_prop])) + jnp.tile(c4_c, (self.NP, 1))
        # e_prop = jnp.tile([jnp.cos(self.tilt_angle), 0, -jnp.sin(self.tilt_angle)], (self.NP, 1))
        # w_prop = jnp.column_stack((b_prop, e_prop, self.D_prop, jnp.tile(rho, (self.NP, 1))))
        # u_prop = self.om_prop
        # x_prop = jnp.tile(jnp.concatenate((uvw, om)), (self.NP, 1))

        # T, T_x, T_om = prop_thrust(x_prop, u_prop, w_prop, self.Ct_prop, ders)
        # Q, Q_x, Q_om = prop_torque(x_prop, u_prop, w_prop, self.Cq_prop, ders)

        # self.Thrust = T
        # self.Torque = Q
        # else: # since hasProp=False
        e_prop = jnp.zeros(3)
        Q = jnp.zeros(1)
        Q_x = jnp.zeros(6)
        Q_om = jnp.zeros(1)
        T_x = jnp.zeros(6)
        T_om = jnp.zeros(1)

        assert self.tilt_angle == 0
        u: Af4 = jnp.column_stack((self.Tki, self.Tli, self.deli, jnp.tile(self.tilt_angle, (self.NS, 1))))
        x: Af6 = jnp.tile(jnp.concatenate((uvw.T, om.T), axis=-1), (self.NS, 1))

        FMi, FMi_x, FMi_u = FM_body(x, u, w, self.aero_coefs_pp, ders)

        Fxi: Af1 = FMi[:, 0]
        Fyi: Af1 = FMi[:, 1]
        Fzi: Af1 = FMi[:, 2]
        Mxi: Af1 = FMi[:, 3]
        Myi: Af1 = FMi[:, 4]
        Mzi: Af1 = FMi[:, 5]

        self.Fx = jnp.sum(Fxi)
        self.Fy = jnp.sum(Fyi)
        self.Fz = jnp.sum(Fzi)
        self.Mx = jnp.sum(Mxi)  # not exact
        self.My = jnp.sum(Myi)  # not exact
        self.Mz = jnp.sum(Mzi)  # not exact

        # Fxi_x = FMi_x[:, :, 0]
        # Fxi_u = FMi_u[:, :, 0]
        # Fyi_x = FMi_x[:, :, 1]
        # Fyi_u = FMi_u[:, :, 1]
        # Fzi_x = FMi_x[:, :, 2]
        # Fzi_u = FMi_u[:, :, 2]
        # Mxi_x = FMi_x[:, :, 3]
        # Mxi_u = FMi_u[:, :, 3]
        # Myi_x = FMi_x[:, :, 4]
        # Myi_u = FMi_u[:, :, 4]
        # Mzi_x = FMi_x[:, :, 5]
        # Mzi_u = FMi_u[:, :, 5]

        # if ders:
        #     dFxi_df = jnp.zeros(self.NS)
        #     dFyi_df = jnp.zeros(self.NS)
        #     dFzi_df = jnp.zeros(self.NS)
        #     dMxi_df = jnp.zeros(self.NS)
        #     dMyi_df = jnp.zeros(self.NS)
        #     dMzi_df = jnp.zeros(self.NS)

        #     dFxi_df = jax.ops.index_add(dFxi_df, self.flap_idx, Fxi_u[self.flap_idx, 2])
        #     dFyi_df = jax.ops.index_add(dFyi_df, self.flap_idx, Fyi_u[self.flap_idx, 2])
        #     dFzi_df = jax.ops.index_add(dFzi_df, self.flap_idx, Fzi_u[self.flap_idx, 2])
        #     dMxi_df = jax.ops.index_add(dMxi_df, self.flap_idx, Mxi_u[self.flap_idx, 2])
        #     dMyi_df = jax.ops.index_add(dMyi_df, self.flap_idx, Myi_u[self.flap_idx, 2])
        #     dMzi_df = jax.ops.index_add(dMzi_df, self.flap_idx, Mzi_u[self.flap_idx, 2])

        #     dFxi_da = jnp.zeros(self.NS)
        #     dFyi_da = jnp.zeros(self.NS)
        #     dFzi_da = jnp.zeros(self.NS)
        #     dMxi_da = jnp.zeros(self.NS)
        #     dMyi_da = jnp.zeros(self.NS)
        #     dMzi_da = jnp.zeros(self.NS)

        #     dFxi_da = jax.ops.index_add(dFxi_da, self.aileron_idx, -self.sgn * Fxi_u[self.aileron_idx, 2])
        #     dFyi_da = jax.ops.index_add(dFyi_da, self.aileron_idx, -self.sgn * Fyi_u[self.aileron_idx, 2])
        #     dFzi_da = jax.ops.index_add(dFzi_da, self.aileron_idx, -self.sgn * Fzi_u[self.aileron_idx, 2])
        #     dMxi_da = jax.ops.index_add(dMxi_da, self.aileron_idx, -self.sgn * Mxi_u[self.aileron_idx, 2])
        #     dMyi_da = jax.ops.index_add(dMyi_da, self.aileron_idx, -self.sgn * Myi_u[self.aileron_idx, 2])
        #     dMzi_da = jax.ops.index_add(dMzi_da, self.aileron_idx, -self.sgn * Mzi_u[self.aileron_idx, 2])

        #     dFxi_di = Fxi_u[:, 3]
        #     dFyi_di = Fyi_u[:, 3]
        #     dFzi_di = Fzi_u[:, 3]
        #     dMxi_di = Mxi_u[:, 3]
        #     dMyi_di = Myi_u[:, 3]
        #     dMzi_di = Mzi_u[:, 3]

        #     Fxi_ub = jnp.column_stack((dFxi_df, dFxi_da, dFxi_di))
        #     Fyi_ub = jnp.column_stack((dFyi_df, dFyi_da, dFyi_di))
        #     Fzi_ub = jnp.column_stack((dFzi_df, dFzi_da, dFzi_di))
        #     Mxi_ub = jnp.column_stack((dMxi_df, dMxi_da, dMxi_di))
        #     Myi_ub = jnp.column_stack((dMyi_df, dMyi_da, dMyi_di))
        #     Mzi_ub = jnp.column_stack((dMzi_df, dMzi_da, dMzi_di))

        #     self.Fx_x = self.Fx_x + jnp.sum(Fxi_x, axis=0)
        #     self.Fy_x = self.Fy_x + jnp.sum(Fyi_x, axis=0)
        #     self.Fz_x = self.Fz_x + jnp.sum(Fzi_x, axis=0)
        #     self.Mx_x = self.Mx_x + jnp.sum(Mxi_x, axis=0) + ex @ Q_x
        #     self.My_x = self.My_x + jnp.sum(Myi_x, axis=0) + ey @ Q_x
        #     self.Mz_x = self.Mz_x + jnp.sum(Mzi_x, axis=0) + ez @ Q_x

        #     self.Fx_u[0, self.NP:] = jnp.sum(Fxi_ub, axis=0)
        #     self.Fy_u[0, self.NP:] = jnp.sum(Fyi_ub, axis=0)
        #     self.Fz_u[0, self.NP:] = jnp.sum(Fzi_ub, axis=0)
        #     self.Mx_u[0, self.NP:] = jnp.sum(Mxi_ub, axis=0) + ex @ Q_om
        #     self.My_u[0, self.NP:] = jnp.sum(Myi_ub, axis=0) + ey @ Q_om
        #     self.Mz_u[0, self.NP:] = jnp.sum(Mzi_ub, axis=0) + ez @ Q_om

        # if self.hasProp:
        #     ex = self.spin * e_prop[:self.NP, 0]
        #     ey = self.spin * e_prop[:self.NP, 1]
        #     ez = self.spin * e_prop[:self.NP, 2]

        #     self.Mx = self.Mx + jnp.sum(ex @ Q[:self.NP, 0])
        #     self.My = self.My + jnp.sum(ey @ Q[:self.NP, 0])
        #     self.Mz = self.Mz + jnp.sum(ez @ Q[:self.NP, 0])

        #     if ders:
        #         dFxi_dT = jnp.zeros((self.NS, self.NP))
        #         dFyi_dT = jnp.zeros((self.NS, self.NP))
        #         dFzi_dT = jnp.zeros((self.NS, self.NP))
        #         dMxi_dT = jnp.zeros((self.NS, self.NP))
        #         dMyi_dT = jnp.zeros((self.NS, self.NP))
        #         dMzi_dT = jnp.zeros((self.NS, self.NP))

        #         for ii in range(self.NP):
        #             idx = (self.k_idx == ii)

        #             dFxi_dT = jax.ops.index_add(dFxi_dT, idx, Fxi_u[idx, 0] * self.Aki[idx] / self.A[ii])
        #             dFyi_dT = jax.ops.index_add(dFyi_dT, idx, Fyi_u[idx, 0] * self.Aki[idx] / self.A[ii])
        #             dFzi_dT = jax.ops.index_add(dFzi_dT, idx, Fzi_u[idx, 0] * self.Aki[idx] / self.A[ii])
        #             dMxi_dT = jax.ops.index_add(dMxi_dT, idx, Mxi_u[idx, 0] * self.Aki[idx] / self.A[ii])
        #             dMyi_dT = jax.ops.index_add(dMyi_dT, idx, Myi_u[idx, 0] * self.Aki[idx] / self.A[ii])
        #             dMzi_dT = jax.ops.index_add(dMzi_dT, idx, Mzi_u[idx, 0] * self.Aki[idx] / self.A[ii])

        #         for ii in range(len(self.Al)):
        #             idx = (self.l_idx == ii)

        #             dTl_dT1 = ((self.A[ii] + self.A[ii + 1] - self.Al[ii]) / (self.A[ii] + self.A[ii + 1])) ** (1 / 3) - (
        #                     self.A[ii] - self.Al[ii]) / self.A[ii]
        #             dTl_dT2 = ((self.A[ii] + self.A[ii + 1] - self.Al[ii]) / (self.A[ii] + self.A[ii + 1])) ** (1 / 3) - (
        #                     self.A[ii + 1] - self.Al[ii]) / self.A[ii + 1]
        #             dTi_dTl = self.Ali[idx] / self.Al[ii]

        #             dFxi_dT = jax.ops.index_add(dFxi_dT, idx, Fxi_u[idx, 1] * dTi_dTl * dTl_dT1)
        #             dFxi_dT = jax.ops.index_add(dFxi_dT, idx, Fxi_u[idx, 1] * dTi_dTl * dTl_dT2)
        #             dFyi_dT = jax.ops.index_add(dFyi_dT, idx, Fyi_u[idx, 1] * dTi_dTl * dTl_dT1)
        #             dFyi_dT = jax.ops.index_add(dFyi_dT, idx, Fyi_u[idx, 1] * dTi_dTl * dTl_dT2)
        #             dFzi_dT = jax.ops.index_add(dFzi_dT, idx, Fzi_u[idx, 1] * dTi_dTl * dTl_dT1)
        #             dFzi_dT = jax.ops.index_add(dFzi_dT, idx, Fzi_u[idx, 1] * dTi_dTl * dTl_dT2)
        #             dMxi_dT = jax.ops.index_add(dMxi_dT, idx, Mxi_u[idx, 1] * dTi_dTl * dTl_dT1)
        #             dMxi_dT = jax.ops.index_add(dMxi_dT, idx, Mxi_u[idx, 1] * dTi_dTl * dTl_dT2)
        #             dMyi_dT = jax.ops.index_add(dMyi_dT, idx, Myi_u[idx, 1] * dTi_dTl * dTl_dT1)
        #             dMyi_dT = jax.ops.index_add(dMyi_dT, idx, Myi_u[idx, 1] * dTi_dTl * dTl_dT2)
        #             dMzi_dT = jax.ops.index_add(dMzi_dT, idx, Mzi_u[idx, 1] * dTi_dTl * dTl_dT1)
        #             dMzi_dT = jax.ops.index_add(dMzi_dT, idx, Mzi_u[idx, 1] * dTi_dTl * dTl_dT2)

        #         dFx_dT = jnp.sum(dFxi_dT, axis=0)
        #         dFy_dT = jnp.sum(dFyi_dT, axis=0)
        #         dFz_dT = jnp.sum(dFzi_dT, axis=0)
        #         dMx_dT = jnp.sum(dMxi_dT, axis=0)
        #         dMy_dT = jnp.sum(dMyi_dT, axis=0)
        #         dMz_dT = jnp.sum(dMzi_dT, axis=0)

        #         dFx_dom = dFx_dT * T_om
        #         dFy_dom = dFy_dT * T_om
        #         dFz_dom = dFz_dT * T_om
        #         dMx_dom = dMx_dT * T_om
        #         dMy_dom = dMy_dT * T_om
        #         dMz_dom = dMz_dT * T_om

        #         self.Fx_x = self.Fx_x + dFx_dT * T_x
        #         self.Fy_x = self.Fy_x + dFy_dT * T_x
        #         self.Fz_x = self.Fz_x + dFz_dT * T_x
        #         self.Mx_x = self.Mx_x + dMx_dT * T_x + ex @ Q_x
        #         self.My_x = self.My_x + dMy_dT * T_x + ey @ Q_x
        #         self.Mz_x = self.Mz_x + dMz_dT * T_x + ez @ Q_x

        #         self.Fx_u[0, :self.NP] = dFx_dom
        #         self.Fy_u[0, :self.NP] = dFy_dom
        #         self.Fz_u[0, :self.NP] = dFz_dom
        #         self.Mx_u[0, :self.NP] = dMx_dom + ex @ Q_om
        #         self.My_u[0, :self.NP] = dMy_dom + ey @ Q_om
        #         self.Mz_u[0, :self.NP] = dMz_dom + ez @ Q_om

        alf = jnp.arctan2(uvw[2], uvw[0])
        self.Li = -Fzi * jnp.cos(alf) + Fxi * jnp.sin(alf)
        self.Di = -Fzi * jnp.sin(alf) - Fxi * jnp.cos(alf)

        return self
