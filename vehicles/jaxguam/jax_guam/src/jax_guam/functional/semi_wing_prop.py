from typing import Literal, NamedTuple

import jax.numpy as jnp
import numpy as np

from jax_guam.classes.base_functions.FM_body import FM_body
from jax_guam.utils.angle_utils import Rx, Ry
from jax_guam.utils.jax_types import (
    Af1,
    Af3,
    Af4,
    Af6,
    Af8,
    AfVec,
    FloatScalar,
    IntScalar,
    Mat1,
    Mat31,
    Vec1,
    Vec2,
    Vec3_1, Vec6, FMVec,
)


class StripsOutput(NamedTuple):
    A: float
    yc: AfVec
    cc: Af1
    Ak: Vec1
    Si: Af1
    Aki: Af1
    Ali: Af1
    k_idx: Af1
    l_idx: Af1
    Tki: AfVec
    Tli: AfVec
    flap_idx: AfVec
    aileron_idx: AfVec
    deli: AfVec


# class SemiWingPropFM(NamedTuple):
#     Fx: FloatScalar
#     Fy: FloatScalar
#     Fz: FloatScalar
#     Mx: FloatScalar
#     My: FloatScalar
#     Mz: FloatScalar


class FuncSemiWingProp:
    """Functional version of SemiWingProp class."""

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
        c4_b: Vec3_1,
    ):
        self.strip_width = 0.2
        self.del_f = 0  # flap deflection angle
        self.del_a = 0  # aileron deflection angle
        self.tau = 0.6  # flap/aileron effectiveness
        self.NS = jnp.floor(span / self.strip_width)
        self.NS = int(self.NS)

        # Note: We don't need to zero-init, this is handled in SetupStrips()
        # self.yc: Af1 = np.zeros((self.NS, 1))
        # self.Si: Af1 = np.zeros((self.NS, 1))
        # self.Aki = np.zeros((self.NS, 1))
        # self.Ali = np.zeros((self.NS, 1))
        # self.k_idx: Af1 = np.zeros((self.NS, 1))
        # self.l_idx = np.zeros((self.NS, 1))
        # # Even after set_thrust, Tki = 0 !
        # self.Tki = np.zeros((self.NS, 1))
        # self.Tli = np.zeros((self.NS, 1))
        # self.flap_idx = np.zeros((self.NS, 1))
        # self.aileron_idx = np.zeros((self.NS, 1))
        # self.deli = np.zeros((self.NS, 1))
        # self.Li = np.zeros((self.NS, 1))
        # self.Di = np.zeros((self.NS, 1))
        # self.alfi = np.zeros((self.NS, 1))

        self.hasProp = False

        # check left or right side
        if lr == "Left":
            self.sgn = -1
        else:
            self.sgn = 1

        # self.airfoil = airfoil  # Note: Unused!
        self.aero_coefs_pp: Mat1 = coeff
        self.b = span
        chord = c
        self.gamma = angle
        self.y_flap = y_flap
        self.y_aileron = y_aileron
        self.h_b: Vec3_1 = c4_b

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
        # self.tilt_angle: int | float = 0

        # Finally set up the strips
        (
            self.A,
            self.yc,
            self.cc,
            self.Ak,
            self.Si,
            self.Aki,
            self.Ali,
            self.k_idx,
            self.l_idx,
            self.Tki,
            self.Tli,
            self.flap_idx,
            self.aileron_idx,
            self.deli_base,
        ) = self.SetupStrips()

        # if self.hasProp:
        #     self.set_thrust(jnp.zeros(self.NP))  # initialize thrust at zero
        #     self.Torque = jnp.zeros(self.NP)
        #     self.om_prop = jnp.zeros((self.NP, 1))
        #     self.e_b = jnp.zeros((3, self.NP))
        # else:
        assert not self.hasProp
        self.Thrust, self.Tki = self.set_thrust(np.zeros(1))
        self.Torque = np.zeros(1)
        self.om_prop = np.zeros(1)
        self.e_b = np.zeros((3, 1))

        # # Note: self.A has already been set to the same value in SetupStrips
        # self.A = jnp.pi * self.D_prop**2 / 4  # propeller disk areas
        self.c4_b: Vec3_1 = self.get_c4_b(tilt_angle=0)

        # self.Fx = 0
        # self.Fy = 0
        # self.Fz = 0
        # self.Mx = 0
        # self.My = 0
        # self.Mz = 0
        #
        # self.Fx_x = jnp.zeros(6)
        # self.Fy_x = jnp.zeros(6)
        # self.Fz_x = jnp.zeros(6)
        # self.Mx_x = jnp.zeros(6)
        # self.My_x = jnp.zeros(6)
        # self.Mz_x = jnp.zeros(6)
        #
        # self.Fx_u = jnp.zeros(self.NP + 3)  # NP + 3 surface inputs
        # self.Fy_u = jnp.zeros(self.NP + 3)
        # self.Fz_u = jnp.zeros(self.NP + 3)
        # self.Mx_u = jnp.zeros(self.NP + 3)
        # self.My_u = jnp.zeros(self.NP + 3)
        # self.Mz_u = jnp.zeros(self.NP + 3)

    def SetupStrips(self) -> StripsOutput:
        """Only called during __init__."""
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
        self_A = A

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
            ci = jnp.concatenate([cw, jnp.zeros((y[y > b].shape[0], 1))], axis=0)
            Si = jnp.concatenate([Sw, jnp.zeros((y[y > b].shape[0], 1))], axis=0)

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

        # Note: dY becomes (1, ) after this, but since we cumsum and then float, its fine to just use it as scalar.
        # dY = jnp.zeros((NA, 1))
        # dY[odd] = D_prop - float(L[:-1] * (L[:-1] > 0) - L[1:] * (L[1:] > 0))
        # dY = dY.at[odd].set(D_prop - float(L[:-1] * (L[:-1] > 0) - L[1:] * (L[1:] > 0))) #Note weird bug
        # dY[even] = jnp.abs(l)
        # Y = float(jnp.cumsum(dY)) + (C_prop - D_prop / 2)
        dY = D_prop - float(L[:-1] * (L[:-1] > 0) - L[1:] * (L[1:] > 0))
        Y = float(dY) + (C_prop - D_prop / 2)

        Di = Y

        zt = np.zeros_like(y)
        k_idx = np.zeros_like(y)
        for i in range(Np, 0, -1):
            # zt[idx] = jnp.sqrt(D_prop**2 / 4 - (y[idx] - C_prop)**2)
            # k_idx[idx] = i

            # # Note: jnp.
            # zt = zt.at[y < Di].set(jnp.sqrt(D_prop**2 / 4 - (y[y < Di] - C_prop) ** 2))
            # k_idx = k_idx.at[y < Di].set(i)

            zt[y < Di] = jnp.sqrt(D_prop**2 / 4 - (y[y < Di] - C_prop) ** 2)
            k_idx[y < Di] = i

        k_idx = k_idx * (zt > 0)
        k_idx = k_idx[:-1]

        zl = np.zeros_like(y)
        l_idx = np.zeros_like(y)
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

        self_yc = y[1:] - np.diff(y) / 2  # y distance from root to center of strip
        self_cc = (ci[:-1] + ci[1:]) / 2  # wing strip center cord lengths

        # self.A = A   # Disk area of each prop
        self_Ak = Ak  # propeller non-overlapped areas
        # self.Al = Al # propeller overlapped areas

        self_Si = Si  # wing strip areas
        self_Aki = np.expand_dims(Aki, axis=1)  # propeller non overlapped strip areas
        self_Ali = np.expand_dims(Ali, axis=1)  # propeller over lapped strip areas

        self_k_idx = np.expand_dims(k_idx, axis=1)
        self_l_idx = np.expand_dims(l_idx, axis=1)

        self_Tki = np.zeros_like(k_idx)
        self_Tli = np.zeros_like(l_idx)

        # Save the flap and aileron indices
        self_flap_idx = (self_yc >= self.y_flap[0]) & (self_yc <= self.y_flap[1])
        self_aileron_idx = (self_yc >= self.y_aileron[0]) & (self_yc <= self.y_aileron[1])
        index = ~(self_flap_idx | self_aileron_idx)

        # mark wing strips with out defelcting surfaces
        # self.deli[index] = 255
        self_deli = np.zeros((self.NS, 1))
        self_deli[index] = 255

        return StripsOutput(
            self_A,
            self_yc,
            self_cc,
            self_Ak,
            self_Si,
            self_Aki,
            self_Ali,
            self_k_idx,
            self_l_idx,
            self_Tki,
            self_Tli,
            self_flap_idx,
            self_aileron_idx,
            self_deli,
        )

    def set_thrust(self, thrust: Vec1) -> tuple[Vec1:Af1]:
        idx = self.k_idx > 0
        T = np.ones(self.k_idx[idx].shape) * thrust
        A = np.ones(self.k_idx[idx].shape) * self.A
        Aki = self.Aki[idx]

        Tki = self.Tki.copy()
        Tki[(idx == 1).reshape(idx.shape[0])] = T * Aki / A
        return thrust, Tki

    def get_c4_b(self, tilt_angle: FloatScalar | IntScalar) -> Vec3_1:
        assert tilt_angle == 0
        r = self.h_b + Ry(tilt_angle) @ self.c4_h
        return r

    def c4_c(self, cm_b: Mat31) -> Mat31:
        return self.c4_b - cm_b

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
        assert isinstance(ders, bool)

        ##########################################################################################
        # Set dela for aileron
        #   get the flap values where they overlap with aileron
        flap = self.flap_idx * self.aileron_idx * self.del_f
        deli = jnp.array(self.deli_base)
        deli = deli.at[self.aileron_idx].set(jnp.expand_dims(flap[self.aileron_idx] - self.sgn * del_a, 1))

        # Set delf for flaps.
        #   get the aileron values where they overlap with flaps
        aileron = self.flap_idx * self.aileron_idx * self.del_a
        deli = deli.at[self.flap_idx].set(jnp.expand_dims(del_f - self.sgn * aileron[self.flap_idx], 1))
        ##########################################################################################

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

        # Note: The values below are set but not used, so we ignore!
        # e_prop = jnp.zeros(3)
        # Q = jnp.zeros(1)
        # Q_x = jnp.zeros(6)
        # Q_om = jnp.zeros(1)
        # T_x = jnp.zeros(6)
        # T_om = jnp.zeros(1)

        assert tilt_angle == 0
        u: Af4 = jnp.column_stack((self.Tki, self.Tli, deli, jnp.tile(tilt_angle, (self.NS, 1))))
        x: Af6 = jnp.tile(jnp.concatenate((uvw.T, om.T), axis=-1), (self.NS, 1))

        FMi, FMi_x, FMi_u = FM_body(x, u, w, self.aero_coefs_pp, ders)

        FM = FMi.sum(axis=0)
        assert FM.shape == (6,)
        return FM
        # Note: Instead of separating out each component, just return the (6, ) FM vector.
        # Fxi: Af1 = FMi[:, 0]
        # Fyi: Af1 = FMi[:, 1]
        # Fzi: Af1 = FMi[:, 2]
        # Mxi: Af1 = FMi[:, 3]
        # Myi: Af1 = FMi[:, 4]
        # Mzi: Af1 = FMi[:, 5]
        #
        # return SemiWingPropFM(Fxi.sum(), Fyi.sum(), Fzi.sum(), Mxi.sum(), Myi.sum(), Mzi.sum())

        # Note: The values below are set but not used, so we ignore!
        # alf = jnp.arctan2(uvw[2], uvw[0])
        # Li = -Fzi * jnp.cos(alf) + Fxi * jnp.sin(alf)
        # Di = -Fzi * jnp.sin(alf) - Fxi * jnp.cos(alf)
