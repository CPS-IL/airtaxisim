#!/usr/bin/env python3

from loguru import logger

import rospy
from trajectory_msgs.msg import JointTrajectoryPoint

import jax.numpy as jnp
from jax_guam.guam_types import RefInputs

from tools.sensors import GuamVelocitySensor

rospy.set_param('freq', 10.0)


def landing_reference_inputs(time, t_landing = 60, ground = 0, initial_height = 100):
    p_init = jnp.array([0, 0, initial_height]) # Initial position
    v_init = jnp.array([0, 0, 0])              # Initial velocity

    # Generate a smooth cubic tragectory for position
    # Height = a*t^3 + b*t^2 + c*t + d
    # Speed = 3*a*t^2 + 2*b*t + c
    d = p_init[2]
    c = v_init[2]
    b = (v_init[2]*t_landing - 3*v_init[2] - 3*p_init[2])/t_landing**2
    a = (-2*b*t_landing - v_init[2])/(3*t_landing**2)

    if time < t_landing:
        pos_des = jnp.array([p_init[0], p_init[1], a*time**3+b*time**2+c*time+d])
        vel_bIc_des = jnp.array([0, 0, 3*a*time**2+2*b*time+c])
        chi_des = 0
        chi_dot_des = 0

    else:
        pos_des = jnp.array([p_init[0], p_init[1], ground])
        vel_bIc_des = jnp.array([0, 0, 0])
        chi_des = 0
        chi_dot_des = 0

    return RefInputs(
        Vel_bIc_des=vel_bIc_des,
        Pos_des=pos_des,
        Chi_des=chi_des,
        Chi_dot_des=chi_dot_des,
    )


class StaticPlanner:

    def __init__(self, z_initial, z_land, time_land):
        rospy.init_node('planner')
        self.rate = rospy.Rate(rospy.get_param('freq'))
        self.pub = rospy.Publisher('/planner/reference', JointTrajectoryPoint, queue_size=1)
        self.index = 0.0
        self.time_delta = 1.0/rospy.get_param('freq')
        self.velocitysensor = GuamVelocitySensor()
        self.initial_pos_z = z_initial
        self.landing_pos_z = z_land
        self.landing_time = time_land
        self.traj_msg = JointTrajectoryPoint()

    def publish(self, ref = None):
        if ref == None:
            # logger.warning("No trajectory set yet.")
            return
        self.traj_msg.positions = ref.Pos_des
        self.traj_msg.velocities = ref.Vel_bIc_des
        self.pub.publish(self.traj_msg)

    def ready(self):
        if not self.velocitysensor.is_ready():
            # logger.warning("GUAM Velocity Sensor not ready")
            return False
        return True

    def tick(self):
        if not self.ready():
            self.rate.sleep()
            return
        self.publish(landing_reference_inputs(
            self.index * self.time_delta, self.landing_time, self.landing_pos_z, self.initial_pos_z))
        self.index = self.index + 1
        self.rate.sleep()


def main():
    planner = StaticPlanner(z_initial = 100, z_land = 0, time_land = 60)
    while not rospy.is_shutdown():
        planner.tick()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
