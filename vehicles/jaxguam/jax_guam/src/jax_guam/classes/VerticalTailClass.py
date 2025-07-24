import ipdb
import jax.numpy as jnp

from jax_guam.classes.SemiWingPropClass import SemiWingPropClass


class VerticalTailClass:
    def __init__(self, airfoil, coeff, span, c, y_rudder, c4_b, mass, I, cm_b):
        airfoil = airfoil
        coeff = coeff
        b = span
        c = c
        y_rudder = y_rudder
        self.c4_b = c4_b

        self.tail = SemiWingPropClass(
            "Right", airfoil, coeff, b, c, jnp.pi / 2, y_rudder, [0, 0], self.c4_b.reshape((3, 1))
        )

        self.mass = mass
        self.I = I
        self.cm_b = cm_b
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

        self.Fx_u = jnp.zeros(1)
        self.Fy_u = jnp.zeros(1)
        self.Fz_u = jnp.zeros(1)
        self.Mx_u = jnp.zeros(1)
        self.My_u = jnp.zeros(1)
        self.Mz_u = jnp.zeros(1)

    def aero(self, rho, uvw, om, cm_b, ders: bool):
        assert isinstance(ders, bool)
        # Call the strip theory aerodynamics
        self.tail = self.tail.aero(rho, uvw, om, cm_b, ders)

        # Assign values from the tail component to the main selfect
        self.Fx = self.tail.Fx
        self.Fy = self.tail.Fy
        self.Fz = self.tail.Fz
        self.Mx = self.tail.Mx
        self.My = self.tail.My
        self.Mz = self.tail.Mz

        # if ders:
        #     self.Fx_x = self.tail.Fx_x
        #     self.Fy_x = self.tail.Fy_x
        #     self.Fz_x = self.tail.Fz_x
        #     self.Mx_x = self.tail.Mx_x
        #     self.My_x = self.tail.My_x
        #     self.Mz_x = self.tail.Mz_x

        #     self.Fx_u = self.tail.Fx_u[:, 0]
        #     self.Fy_u = self.tail.Fy_u[:, 0]
        #     self.Fz_u = self.tail.Fz_u[:, 0]
        #     self.Mx_u = self.tail.Mx_u[:, 0]
        #     self.My_u = self.tail.My_u[:, 0]
        #     self.Mz_u = self.tail.Mz_u[:, 0]
        return self
