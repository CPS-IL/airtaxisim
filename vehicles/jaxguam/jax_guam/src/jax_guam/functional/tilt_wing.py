import pdb
from typing import NamedTuple

import ipdb
import jax.numpy as jnp
import numpy as np
from jax import jit

from jax_guam.classes.MassClass import MassClass
from jax_guam.classes.PropellerClass import PropellerClass
from jax_guam.functional.body import FuncBody
from jax_guam.functional.propeller import FuncPropeller
from jax_guam.functional.tail import FuncTail
from jax_guam.functional.wing_prop import FuncWingProp
from jax_guam.guam_types import FM, AeroFM, PropFM
from jax_guam.utils.jax_types import FloatScalar, FMVec, IntScalar, PropVec, Vec3_1, Vec9_1


class TiltwingControls(NamedTuple):
    # Propeller speeds? 8 for lift, then 1 pusher?
    om_prop: Vec9_1
    # Aileron
    del_a: FloatScalar
    # Flaps
    del_f: FloatScalar
    # Elevator
    del_e: FloatScalar
    # Rudder
    del_r: FloatScalar
    # Wing Tilt Angle
    i_w: FloatScalar | IntScalar
    # Tail Tilt Angle
    i_t: FloatScalar | IntScalar


class FuncTiltWing:
    def __init__(
        self,
        WingProp: FuncWingProp,
        tail: FuncTail,
        body: FuncBody,
        props: list[FuncPropeller],
        Extra_Mass: list[MassClass],
    ):
        self.wingprop = WingProp
        self.tail = tail
        self.body = body
        self.props = props

        assert tail is not None
        assert body is not None
        assert len(props) > 0

        # self.om_p = jnp.zeros(len(props))  # Note: Passed in aero instead
        self.Masses = Extra_Mass

        self.cm_b: Vec3_1 = self.get_cm_b()

    @property
    def n_props(self) -> int:
        return len(self.props)

    def get_mass(self):
        r = self.wingprop.mass + self.tail.mass
        if self.props:
            for i in range(len(self.props)):
                r += self.props[i].mass
        if self.body:
            r += self.body.mass
        if self.Masses:
            for i in range(len(self.Masses)):
                r += self.Masses[i].mass
        return r

    def get_cm_b(self) -> Vec3_1:
        w_cm_b = self.wingprop.cm_b
        t_cm_b = self.tail.cm_b
        w_m = self.wingprop.mass
        t_m = self.tail.mass

        numProps = len(self.props)
        p_cm_b = np.zeros((3, numProps))
        p_mass = np.zeros((numProps, 1))
        if not self.props:
            raise ValueError("Should have Prop!")
            # p_cm_b = jnp.array([[0.0], [0.0], [0.0]])
            # p_m = 0.0
        else:
            for ii in range(numProps):
                # p_cm_b[:, ii] = self.Prop[ii].cm_b
                # p_mass[ii] = self.Prop[ii].mass

                # Note: jnp version
                # p_cm_b = p_cm_b.T.at[ii].set(self.Prop[ii].cm_b).T
                # p_mass = p_mass.at[ii].set(self.Prop[ii].mass)  # .reshape(numProps) #Note weird

                p_cm_b[:, ii] = self.props[ii].cm_b
                p_mass[ii] = self.props[ii].mass  # .reshape(numProps) #Note weird

            p_mass = p_mass.reshape(numProps)
            p_m = p_mass * jnp.eye(9)

        if not self.body:
            raise ValueError("Should have Prop!")
            # b_cm_b = jnp.array([[0.0], [0.0], [0.0]])
            # b_m = 0.0
        else:
            b_cm_b = self.body.cm_b
            b_m = self.body.mass

        # include extra masses if present
        numMasses = len(self.Masses)
        m_cm_b = np.zeros((3, numMasses))
        m_mass = np.zeros((numMasses, 1))
        if not self.Masses:
            m_cm_b = jnp.array([[0.0], [0.0], [0.0]])
            m_m = 0.0
        else:
            for ii in range(numMasses):
                # m_cm_b[:, ii] = self.Masses[ii].cm_b
                # m_mass[ii] = self.Masses[ii].mass
                # import pdb; pdb.set_trace()

                # # Note: jnp version
                # m_cm_b = m_cm_b.T.at[ii].set(self.Masses[ii].cm_b.reshape(3)).T
                # m_mass = m_mass.at[ii].set(self.Masses[ii].mass)  # Note weird

                m_cm_b[:, ii] = self.Masses[ii].cm_b.squeeze(-1)
                m_mass[ii] = self.Masses[ii].mass

            m_mass = m_mass.reshape(numMasses)
            m_m = jnp.diag(m_mass)

        # compute center of gravity in the body frame
        #   1: Get the point masses for each component.
        #       Each is (3, 1)
        masses = np.stack(
            [
                w_cm_b * w_m,
                t_cm_b * t_m,
                np.sum(p_cm_b @ p_m, axis=1).reshape((3, 1)),
                b_cm_b * b_m,
                np.sum(m_cm_b @ m_m, axis=1).reshape((3, 1)),
            ],
        )
        assert masses.shape == (5, 3, 1)

        mass = self.get_mass()
        r = (1.0 / mass) * np.sum(masses, axis=0)
        return r

    def aero(
        self, rho: float, uvw: Vec3_1, om: Vec3_1, ders: bool, controls: TiltwingControls
    ) -> tuple[FMVec, AeroFM, PropFM]:
        assert isinstance(ders, bool) and ders is False
        fm_wingprop = self.wingprop.aero(rho, uvw, om, self.cm_b, ders, controls.del_a, controls.del_f, controls.i_w)
        fm_tail = self.tail.aero(rho, uvw, om, self.cm_b, ders, controls.del_e, controls.del_r, controls.i_t)
        fms = [fm_wingprop, fm_tail]

        if self.body is not None:
            fm_body = self.body.aero(rho, uvw, om, self.cm_b, ders)
            fms.append(fm_body)

        aero_fms = jnp.stack(fms, axis=0)
        aero_fm = jnp.sum(aero_fms, axis=0)
        assert aero_fm.shape == (6,)

        prop_fms = []
        Ts, Qs = [], []
        assert len(self.props) > 0
        prop: FuncPropeller
        for ii, prop in enumerate(self.props):
            fm_prop, T, Q = prop.aero(rho, uvw, om, self.cm_b, ders, controls.om_prop[ii].squeeze(-1))
            prop_fms.append(fm_prop)
            Ts.append(T)
            Qs.append(Q)

        Tp = jnp.stack(Ts, axis=0).squeeze(-1).squeeze(-1)
        Qp = jnp.stack(Qs, axis=0).squeeze(-1).squeeze(-1)
        assert self.n_props == 9
        assert Tp.shape == Qp.shape == (self.n_props,)

        prop_fms = jnp.stack(prop_fms, axis=0)
        prop_fm = jnp.sum(prop_fms, axis=0)
        assert prop_fm.shape == (6,)

        # Note: Unused
        # alf = jnp.arctan2(uvw[2], uvw[0])
        # self.L = -self.Fz * jnp.cos(alf) + self.Fx * jnp.sin(alf)
        # self.D = -self.Fz * jnp.sin(alf) - self.Fx * jnp.cos(alf)

        total_FM = aero_fm + prop_fm
        aero_FM = AeroFM(aero_fm[:3, None], aero_fm[3:, None])
        prop_FM = PropFM(prop_fm[:3, None], prop_fm[3:, None], Tp, Qp)

        return total_FM, aero_FM, prop_FM
