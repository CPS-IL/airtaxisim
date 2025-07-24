#!/usr/bin/env python3

import argparse
import atexit
import carla
import glob
import numpy as np
import os
import rospy
import signal
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from tools.environment import Environment

from loguru import logger as log
sys.path.append(os.path.abspath('/catkin_ws/src/env_sim/utils'))
from utils.config import load_yaml_file
from utils import constants

FREQ_LOW_LEVEL = 10

class GracefulShutdown:
    def __init__(self, environment):
        self.environment = environment
        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        self.environment.destroy()
        log.info("Carla environment destroyed.")


def run_carla_node(args, client):
    pygame.init()
    config = load_yaml_file(constants.merged_config_path, __file__)

    # rosnode node initialization
    rospy.init_node('carla_node')
    rospy.set_param('tracking_control', False)

    environment = Environment(args, client, config)
    _ = GracefulShutdown(environment)


    rospy.set_param('reset_called', False)
    rospy.set_param('episode_done', False)
    rospy.set_param('done_ack', False)

    try:

        # Running rate
        rate=rospy.Rate(FREQ_LOW_LEVEL)


        #Simulation loop
        call_exit = False

        while not rospy.is_shutdown():

            environment.client_clock.tick_busy_loop(FREQ_LOW_LEVEL)

            # Carla Tick
            environment.tick() # <---- It includes ROS massage subscription and publishing.
            rate.sleep()

            ##########################
            ### Handle Reset Calls ###
            ##########################

            # Reset Call from High-Level Decision Maker that determine both termination and rewards
            if not(rospy.get_param('done_ack')) and rospy.get_param('episode_done'):
                environment.reset()
                rospy.set_param('done_ack', True)
            elif rospy.get_param('done_ack') and not(rospy.get_param('episode_done')):
                rospy.set_param('done_ack', False)

            # Reset Call from Input Display.
            #reset_called = rospy.get_param('reset_called')
            if rospy.get_param('reset_called'):
                print('HERE????')
                environment.reset()
                reset_called = False
                rospy.set_param('reset_ack', True)
            else:
                rospy.set_param('reset_ack', False)

    finally:
        environment.destroy()

def main():
    argparser = argparse.ArgumentParser(
        description='ROS CARLA NODE')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x400',
        help='window resolution (default: 800x400)')
    argparser.add_argument(
        '--asynch',
        action='store_false',
        help='Activate asynchronous mode execution')


    ### Traffic Setting ###
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=0,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')

    args, unknown = argparser.parse_known_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.asynch = False

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        run_carla_node(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        pass