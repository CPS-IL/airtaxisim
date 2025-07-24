#!/usr/bin/env python3

import functools as ft
import ipdb
import numpy as np
import os
import rospy

from loguru import logger

from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

import jax
import jax.random as jr
import jax.numpy as jnp
import jax.tree_util as jtu
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.guam_types import RefInputs
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format

from guam_plot_batch_with_ref import plot_batch_with_ref

from utils.vehicle import Vehicle_Node
from utils.config import load_yaml_file
from utils import constants

class GUAM_Node(Vehicle_Node):
    def __init__(self, config):
        self.config = config
        super(GUAM_Node, self).__init__(config)

        jax_use_cpu()
        jax_use_double()
        set_logger_format()

        # sim params
        self.skip_sleep = config['ego_vehicle']['skip_sleep']
        self.plot_switch = config['ego_vehicle']['plot']
        self.save_video = config['ego_vehicle']['save_video']

        # jax-guam related params
        self.guam = None
        self.guam_reference = None
        self.guam_reference_sub = None

        logger.info("Subscribing to planner for trajectory reference...")
        self.guam_reference_sub = rospy.Subscriber(config['ego_vehicle']['planner_topic'],
                                                    Float32MultiArray,
                                                    self.guam_reference_callback)
        self.guam_reference_init()
        while self.guam_reference is None:
            pass

    def guam_reference_init(self):
        # adjust for guam units and frame
        pos_des = jnp.array(self.initial_position) * jnp.array([3.28084, 3.28084, -3.28084])
        vel_bIc_des = jnp.array(self.initial_velocity) * jnp.array([3.28084, -3.28084, -3.28084])
        chi_des = 0
        chi_dot_des = 0
        self.guam_reference = RefInputs(
            Vel_bIc_des=vel_bIc_des,
            Pos_des=pos_des,
            Chi_des=chi_des,
            Chi_dot_des=chi_dot_des,
        )

    def guam_reference_callback(self, msg):
        # convert meter to feet
        # pos_des = jnp.array([msg.linear.x, msg.linear.y, msg.linear.z]) * jnp.array([3.28084, 3.28084, -3.28084])
        # vel_bIc_des = jnp.array(self.guam_reference.Vel_bIc_des) * jnp.array([3.28084, -3.28084, -3.28084])
        pos_des = jnp.array([msg.data[0], msg.data[1], msg.data[2]]) * jnp.array([3.28084, 3.28084, -3.28084])
        vel_bIc_des = jnp.array([msg.data[3], msg.data[4], msg.data[5]]) * jnp.array([3.28084, -3.28084, -3.28084])
        chi_des = 0
        chi_dot_des = 0
        self.guam_reference = RefInputs(
            Vel_bIc_des=vel_bIc_des,
            Pos_des=pos_des,
            Chi_des=chi_des,
            Chi_dot_des=chi_dot_des,
        )

    def main(self):
        logger.info("Constructing GUAM...")
        self.guam = FuncGUAM()
        batch_size = 1 # Simulate for a single vehicle test
        state = GuamState.create()

        # Assign first value of traj reference as initial guam state
        # [   0:3  ,   3:6    ,   6:9  ,  9:13 ]
        # [ vel_bEb, Omega_BIb, pos_bii, Q_i2b ]
        # state.aircraft[:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, -100*3.28084, 1, 0, 0, 0]) # Initial pos is z = -100 # Q1 must be 1
        # state.aircraft[:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, -0*3.28084, 1, 0, 0, 0]) # Initial pos is z = -100 # Q1 must be 1
        state.aircraft[:] = np.array([self.guam_reference.Vel_bIc_des[0],
                                      self.guam_reference.Vel_bIc_des[1],
                                      self.guam_reference.Vel_bIc_des[2],
                                      0, 0, 0,
                                      self.guam_reference.Pos_des[0],
                                      self.guam_reference.Pos_des[1],
                                      self.guam_reference.Pos_des[2],
                                      1, 0, 0, 0]) # Q1 must be 1

        logger.info("Calling GUAM...")
        b_state: GuamState = jtu.tree_map(lambda x: np.broadcast_to(x, (batch_size,) + x.shape).copy(), state)

        # Perturb the initial state in the x and y directions.
        # key0, key1 = jr.split(jr.PRNGKey(0))
        # b_state.aircraft[:, 6] = jr.uniform(key0, (batch_size,), minval=-20.0, maxval=20.0)
        # b_state.aircraft[:, 7] = jr.uniform(key1, (batch_size,), minval=-20.0, maxval=20.0)

        vmap_step = jax.jit(jax.vmap(ft.partial(self.guam.step, self.guam.dt), in_axes=(0, None)))
        loop_rate = rospy.Rate(1/self.guam.dt) #in Hz guam.dt = 0.005, so rate = 200Hz

        def simulate_batch(self, b_state0) -> GuamState:
            b_state = b_state0
            if self.plot_switch:
                Tb_state = [b_state0]
                time_list = [0]
                vel_des0 = b_state0.aircraft[0][0:3].tolist()
                pos_des0 = b_state0.aircraft[0][6:9].tolist()
                Ref_list = [vel_des0 + pos_des0]

            kk = 0
            while not rospy.is_shutdown():
                t = kk * self.guam.dt
                kk = kk + 1

                ref_inputs = self.guam_reference
                b_state = vmap_step(b_state, ref_inputs)

                if self.plot_switch:
                    time_list.append(t)
                    Ref_list.append(ref_inputs.Vel_bIc_des.tolist() + ref_inputs.Pos_des.tolist()) # Store vel and pos reference
                    Tb_state.append(jax2np(b_state))

                # Assign guam state to publisher
                # [   0:3  ,   3:6    ,   6:9  ,  9:13 ]
                # [ vel_bEb, Omega_BIb, pos_bii, Q_i2b ]
                # Convert from feet to meter
                # North East Down in Guam ==> North East Up
                self.vehicle_pose_msg.pose.position.x = b_state.aircraft[0][6] / 3.28084            # North
                self.vehicle_pose_msg.pose.position.y = b_state.aircraft[0][7] / 3.28084
                self.vehicle_pose_msg.pose.position.z = b_state.aircraft[0][8] / 3.28084 * -1
                
                # The lines below account for the frame of reference differences
                # The second chunk of lines below applies a necessary rotation to print coordinates in CARLA's global frame of reference
                q_x = b_state.aircraft[0][9]
                q_y = b_state.aircraft[0][10]
                q_z = b_state.aircraft[0][11]
                q_w = b_state.aircraft[0][12]
                self.vehicle_pose_msg.pose.orientation.x = -q_w
                self.vehicle_pose_msg.pose.orientation.y = -q_z
                self.vehicle_pose_msg.pose.orientation.z = q_y
                self.vehicle_pose_msg.pose.orientation.w = q_x
                self.vehicle_pose_pub.publish(self.vehicle_pose_msg)

                self.vehicle_vel_msg.linear.x = b_state.aircraft[0][0] / 3.28084            # North
                self.vehicle_vel_msg.linear.y = b_state.aircraft[0][1] / 3.28084
                self.vehicle_vel_msg.linear.z = b_state.aircraft[0][2] / 3.28084 * -1
                self.vehicle_vel_pub.publish(self.vehicle_vel_msg)

                if self.skip_sleep == False:
                    loop_rate.sleep()

            if self.plot_switch:
                bT_state = jtu.tree_map(lambda *args: np.stack(list(args), axis=1), *Tb_state)
                return bT_state, Ref_list, time_list

        if self.plot_switch:
            if not os.path.exists("results"):
                os.makedirs("results")
            bT_state, Ref_list, time_list = simulate_batch(self, b_state)
            np.savez("results/bT_state.npz", aircraft=bT_state.aircraft)
            Ref_list = np.array(Ref_list)
            np.savez("results/Ref_list.npz", Vel_des=Ref_list[:, 0:3], Pos_des=Ref_list[:, 3:6])
            np.savez("results/time_list.npz", np.array(time_list))
        else:
            simulate_batch(self, b_state)

if __name__ == "__main__":
    config = load_yaml_file(constants.merged_config_path, __file__)

    vehicle_type = config['ego_vehicle']['type']
    assert vehicle_type == 'jaxguam', "This node only supports JaxGUAM vehicle, remove jaxguam service from config."

    if config['ego_vehicle']['debug']:
        with ipdb.launch_ipdb_on_exception():
            guam_node = GUAM_Node(config)
            guam_node.main()
    else:
        guam_node = GUAM_Node(config)
        guam_node.main()

    if vehicle_type == 'guam' and config['ego_vehicle']['plot']:
        logger.info("Ploting...")
        plot_batch_with_ref(save_video=config['ego_vehicle']['save_video'])
