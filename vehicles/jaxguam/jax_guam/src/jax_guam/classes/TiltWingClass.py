import pdb

import ipdb
import jax.numpy as jnp
from jax import jit

from jax_guam.classes.BodyClass import BodyClass
from jax_guam.classes.PropellerClass import PropellerClass
from jax_guam.classes.TailClass import TailClass
from jax_guam.classes.WingPropClass import WingPropClass
from jax_guam.utils.jax_types import Vec3_1


class TiltWingClass:
    def __init__(self, WingProp: WingPropClass, Tail: TailClass, Body: BodyClass, Prop: list[PropellerClass], Extra_Mass):
        self.WingProp = WingProp
        self.Tail = Tail
        self.Body = Body
        self.Prop = Prop

        self.om_p = jnp.zeros(len(Prop))
        self.Masses = Extra_Mass
        self.L = 0.0
        self.D = 0.0
        self.Fx = 0.0
        self.Fy = 0.0
        self.Fz = 0.0
        self.Mx = 0.0
        self.My = 0.0
        self.Mz = 0.0
        self.Fx_x = jnp.zeros(6)
        self.Fy_x = jnp.zeros(6)
        self.Fz_x = jnp.zeros(6)
        self.Mx_x = jnp.zeros(6)
        self.My_x = jnp.zeros(6)
        self.Mz_x = jnp.zeros(6)
        self.Fx_u = jnp.zeros(42)
        self.Fy_u = jnp.zeros(42)
        self.Fz_u = jnp.zeros(42)
        self.Mx_u = jnp.zeros(42)
        self.My_u = jnp.zeros(42)
        self.Mz_u = jnp.zeros(42)
        self.aero_Fx = 0.0
        self.aero_Fy = 0.0
        self.aero_Fz = 0.0
        self.aero_Mx = 0.0
        self.aero_My = 0.0
        self.aero_Mz = 0.0

        self.prop_Fx = 0.0
        self.prop_Fy = 0.0
        self.prop_Fz = 0.0
        self.prop_Mx = 0.0
        self.prop_My = 0.0
        self.prop_Mz = 0.0

        self.cm_b: Vec3_1 = self.get_cm_b()
        self.set_del_a(0.0)
        self.set_del_f(0.0)
        self.set_del_e(0.0)
        self.set_del_r(0.0)
        self.set_i_w(0)
        self.set_i_t(0)

        self.mass = None

    def total_Fb(self):
        # Forces in the wind frame
        return jnp.array([self.Fx, self.Fy, self.Fz])

    def total_Mb(self):
        # Moments in the wind frame
        return jnp.array([self.Mx, self.My, self.Mz])

    # Aerodynamic forces/moments
    def aero_Fb(self):
        return jnp.array([[self.aero_Fx], [self.aero_Fy], [self.aero_Fz]])

    def aero_Mb(self):
        return jnp.array([[self.aero_Mx], [self.aero_My], [self.aero_Mz]])

    def prop_Fb(self):
        return jnp.array([self.prop_Fx, self.prop_Fy, self.prop_Fz])

    def prop_Mb(self):
        return jnp.array([self.prop_Mx, self.prop_My, self.prop_Mz])

    def get_Tp(self):
        # omit om_p empty
        numProps = len(self.Prop)
        r = jnp.zeros((1, numProps))
        for i in range(numProps):
            # Note: I think this is an error in the indexing?
            # r = r.at[i].set(self.Prop[i].T)
            r = r.at[0, i].set(self.Prop[i].T)
        # self.Tp = r # Note: Unused
        return r

    def get_Qp(self):
        # omit om_p empty
        numProps = len(self.Prop)
        r = jnp.zeros((1, numProps))
        for i in range(numProps):
            # Note: I think this is an error in the indexing?
            # r = r.at[i].set(self.Prop[i].Q)
            r = r.at[0, i].set(self.Prop[i].Q)
        # self.Qp = r # Note: Unused
        return r

    def get_cm_b(self) -> Vec3_1:
        w_cm_b = self.WingProp.cm_b
        t_cm_b = self.Tail.cm_b
        w_m = self.WingProp.mass
        t_m = self.Tail.mass

        numProps = len(self.Prop)
        p_cm_b = jnp.zeros((3, numProps))
        p_mass = jnp.zeros((numProps, 1))
        if not self.Prop:
            p_cm_b = jnp.array([[0.0], [0.0], [0.0]])
            p_m = 0.0
        else:
            for ii in range(numProps):
                # p_cm_b[:, ii] = self.Prop[ii].cm_b
                # p_mass[ii] = self.Prop[ii].mass
                p_cm_b = p_cm_b.T.at[ii].set(self.Prop[ii].cm_b).T
                p_mass = p_mass.at[ii].set(self.Prop[ii].mass)  # .reshape(numProps) #Note weird
            p_mass = p_mass.reshape(numProps)
            p_m = p_mass * jnp.eye(9)

        if not self.Body:
            b_cm_b = jnp.array([[0.0], [0.0], [0.0]])
            b_m = 0.0
        else:
            b_cm_b = self.Body.cm_b
            b_m = self.Body.mass

        # include extra masses if present
        numMasses = len(self.Masses)
        m_cm_b = jnp.zeros((3, numMasses))
        m_mass = jnp.zeros((numMasses, 1))
        if not self.Masses:
            m_cm_b = jnp.array([[0.0], [0.0], [0.0]])
            m_m = 0.0
        else:
            for ii in range(numMasses):
                # m_cm_b[:, ii] = self.Masses[ii].cm_b
                # m_mass[ii] = self.Masses[ii].mass
                # import pdb; pdb.set_trace()
                m_cm_b = m_cm_b.T.at[ii].set(self.Masses[ii].cm_b.reshape(3)).T
                m_mass = m_mass.at[ii].set(self.Masses[ii].mass)  # Note weird
            m_mass = m_mass.reshape(numMasses)
            m_m = jnp.diag(m_mass)

        # compute center of gravity in the body frame
        self.mass = self.get_mass()
        r: Vec3_1 = (1.0 / self.mass) * jnp.sum(
            jnp.array(
                [
                    w_cm_b * w_m,
                    t_cm_b * t_m,
                    jnp.sum(p_cm_b @ p_m, axis=1).reshape((3, 1)),
                    b_cm_b * b_m,
                    jnp.sum(m_cm_b @ m_m, axis=1).reshape((3, 1)),
                ]
            ),
            axis=0,
        )

        return r

    def get_mass(self):
        r = self.WingProp.mass + self.Tail.mass
        if self.Prop:
            for i in range(len(self.Prop)):
                r += self.Prop[i].mass
        if self.Body:
            r += self.Body.mass
        if self.Masses:
            for i in range(len(self.Masses)):
                r += self.Masses[i].mass
        return r

    def set_om_p(self, om_prop):
        self.om_p = om_prop
        for i in range(len(self.Prop)):
            self.Prop[i].om_prop = om_prop[i][0]

    def set_del_a(self, del_a):
        if abs(del_a) < 30 * jnp.pi / 180:
            self.del_a = del_a
            self.WingProp.set_del_a(del_a)

    def set_del_f(self, del_f):
        if abs(del_f) < 30 * jnp.pi / 180:
            self.del_f = del_f
            self.WingProp.set_del_f(del_f)

    def set_del_e(self, del_e):
        if abs(del_e) < 30 * jnp.pi / 180:
            self.del_e = del_e
            self.Tail.set_del_e(del_e)

    def set_del_r(self, del_r):
        if abs(del_r) < 30 * jnp.pi / 180:
            self.del_r = del_r
            self.Tail.set_del_r(del_r)

    def set_i_w(self, i_w):
        assert i_w == 0
        if i_w <= jnp.pi / 2 and i_w >= 0:
            self.i_w = i_w
            self.WingProp.set_tilt_angle(i_w)

    def set_i_t(self, i_t: float) -> None:
        if i_t <= jnp.pi / 2 and i_t >= 0:
            self.i_t = i_t
            self.Tail.set_tilt_angle(i_t)

    def aero(self, rho, uvw, om, ders: bool):
        assert isinstance(ders, bool)
        self.WingProp = self.WingProp.aero(rho, uvw, om, self.cm_b, ders)
        self.Tail = self.Tail.aero(rho, uvw, om, self.cm_b, ders)

        self.Fx = self.WingProp.Fx + self.Tail.Fx
        self.Fy = self.WingProp.Fy + self.Tail.Fy
        self.Fz = self.WingProp.Fz + self.Tail.Fz
        self.Mx = self.WingProp.Mx + self.Tail.Mx
        self.My = self.WingProp.My + self.Tail.My
        self.Mz = self.WingProp.Mz + self.Tail.Mz

        # if ders:
        #     self.Fx_x = self.WingProp.Fx_x + self.Tail.Fx_x
        #     self.Fy_x = self.WingProp.Fy_x + self.Tail.Fy_x
        #     self.Fz_x = self.WingProp.Fz_x + self.Tail.Fz_x
        #     self.Mx_x = self.WingProp.Mx_x + self.Tail.Mx_x
        #     self.My_x = self.WingProp.My_x + self.Tail.My_x
        #     self.Mz_x = self.WingProp.Mz_x + self.Tail.Mz_x

        #     self.Fx_u = jnp.concatenate([self.WingProp.Fx_u, self.Tail.Fx_u])
        #     self.Fy_u = jnp.concatenate([self.WingProp.Fy_u, self.Tail.Fy_u])
        #     self.Fz_u = jnp.concatenate([self.WingProp.Fz_u, self.Tail.Fz_u])
        #     self.Mx_u = jnp.concatenate([self.WingProp.Mx_u, self.Tail.Mx_u])
        #     self.My_u = jnp.concatenate([self.WingProp.My_u, self.Tail.My_u])
        #     self.Mz_u = jnp.concatenate([self.WingProp.Mz_u, self.Tail.Mz_u])

        if self.Body is not None:
            self.Body = self.Body.aero(rho, uvw, om, self.cm_b, ders)
            self.Fx = self.Fx + self.Body.Fx
            self.Fy = self.Fy + self.Body.Fy
            self.Fz = self.Fz + self.Body.Fz
            self.Mx = self.Mx + self.Body.Mx
            self.My = self.My + self.Body.My
            self.Mz = self.Mz + self.Body.Mz

            # if ders:
            #     self.Fx_x = self.Fx_x + self.Body.Fx_x
            #     self.Fy_x = self.Fy_x + self.Body.Fy_x
            #     self.Fz_x = self.Fz_x + self.Body.Fz_x
            #     self.Mx_x = self.Mx_x + self.Body.Mx_x
            #     self.My_x = self.My_x + self.Body.My_x
            #     self.Mz_x = self.Mz_x + self.Body.Mz_x

        self.aero_Fx = self.Fx
        self.aero_Fy = self.Fy
        self.aero_Fz = self.Fz
        self.aero_Mx = self.Mx
        self.aero_My = self.My
        self.aero_Mz = self.Mz

        self.prop_Fx = 0.0
        self.prop_Fy = 0.0
        self.prop_Fz = 0.0
        self.prop_Mx = 0.0
        self.prop_My = 0.0
        self.prop_Mz = 0.0

        for ii in range(len(self.Prop)):
            self.Prop[ii] = self.Prop[ii].aero(rho, uvw, om, self.cm_b, ders)

            self.prop_Fx = self.prop_Fx + self.Prop[ii].Fx
            self.prop_Fy = self.prop_Fy + self.Prop[ii].Fy
            self.prop_Fz = self.prop_Fz + self.Prop[ii].Fz
            self.prop_Mx = self.prop_Mx + self.Prop[ii].Mx
            self.prop_My = self.prop_My + self.Prop[ii].My
            self.prop_Mz = self.prop_Mz + self.Prop[ii].Mz
            # import pdb; pdb.set_trace()

            # if ders:
            #     self.Fx_x = self.Fx_x + self.Prop[ii].Fx_x
            #     self.Fy_x = self.Fy_x + self.Prop[ii].Fy_x
            #     self.Fz_x = self.Fz_x + self.Prop[ii].Fz_x
            #     self.Mx_x = self.Mx_x + self.Prop[ii].Mx_x
            #     self.My_x = self.My_x + self.Prop[ii].My_x
            #     self.Mz_x = self.Mz_x + self.Prop[ii].Mz_x
            #     #missing something

        self.Fx = self.Fx + self.prop_Fx
        self.Fy = self.Fy + self.prop_Fy
        self.Fz = self.Fz + self.prop_Fz
        self.Mx = self.Mx + self.prop_Mx
        self.My = self.My + self.prop_My
        self.Mz = self.Mz + self.prop_Mz

        Fb = jnp.array([self.Fx, self.Fy, self.Fz])
        Mb = jnp.array([self.Mx, self.My, self.Mz])

        alf = jnp.arctan2(uvw[2], uvw[0])
        self.L = -self.Fz * jnp.cos(alf) + self.Fx * jnp.sin(alf)
        self.D = -self.Fz * jnp.sin(alf) - self.Fx * jnp.cos(alf)
        return self
