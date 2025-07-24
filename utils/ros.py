import os
import roslaunch
import rospy
import subprocess
import time

from loguru import logger as log


class ROSManager:
    def __init__(self, workspace_path):
        self.workspace_path = workspace_path
        self.roscore_process = None
        self.node_processes = {}
        self.launch = None
        self.start_roscore()

    def build_workspace(self):
        log.info(f"Building ROS workspace: {self.workspace_path}")
        os.chdir(self.workspace_path)

        try:
            subprocess.check_call(["catkin_make"])
            log.info("Workspace built successfully.")
        except subprocess.CalledProcessError as e:
            log.error(f"Build failed: {e}")
            return False
        return True

    def start_roscore(self):
        """Start roscore in a background process."""
        if self.roscore_process is None:
            log.info("Starting ROS Master (roscore)...")
            self.roscore_process = subprocess.Popen(['roscore'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)  # Give some time for roscore to initialize
        else:
            log.info("ROS Master is already running.")
    
    def stop_roscore(self):
        """Stop the running ROS Master (roscore)."""
        if self.roscore_process:
            log.info("Stopping ROS Master (roscore)...")
            self.roscore_process.terminate()
            self.roscore_process.wait()
            self.roscore_process = None
        else:
            log.info("ROS Master is not running.")

    def start_launch_file(self, package_name, launch_file_name):
        log.info(f"Launching ROS nodes from {launch_file_name} in package {package_name}...")
        
        # Generate a UUID for the roslaunch session
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        
        # Define the path to the launch file
        launch_file_path = [roslaunch.rlutil.resolve_launch_arguments([package_name, launch_file_name])[0]]
        
        # Create a ROSLaunchParent object to manage the launch file
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, launch_file_path)
        
        # Start the launch file
        self.launch.start()
        rospy.loginfo("Launch file started.")
        log.info("Waiting for Ctrl+C or simulation end.")
        self.launch.spin()        

    def stop_launch_file(self):
        """Stop the currently running launch file."""
        if self.launch:
            rospy.loginfo("Shutting down ROS launch...")
            self.launch.shutdown()
            self.launch = None
        else:
            rospy.logwarn("No launch file is running.")

    def start_node(self, package, executable, node_name=None, args=None):
        """Start a specific ROS node."""
        log.info(f"Starting node: {executable} from package: {package}")
        if args is None:
            args = []

        node = roslaunch.core.Node(package, executable, name=node_name, args=" ".join(args))

        if self.launch == None:
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            self.launch = roslaunch.scriptapi.ROSLaunch()
            self.launch.start()

        process = self.launch.launch(node)
        self.node_processes[node_name or executable] = process
        return process

    def restart_node(self, package, executable, node_name=None, args=None):
        """Restart a specific ROS node."""
        log.info(f"Restarting node: {executable} from package: {package}")
        self.stop_node(node_name or executable)
        time.sleep(1)  # Give a small delay before restarting
        self.start_node(package, executable, node_name, args)

    def stop_node(self, node_name):
        """Stop a specific ROS node."""
        log.info(f"Stopping node: {node_name}")
        process = self.node_processes.get(node_name)
        if process and process.is_alive():
            process.stop()
            self.node_processes.pop(node_name, None)

    def monitor_node(self, node_name):
        """Monitor if a specific ROS node is running."""
        process = self.node_processes.get(node_name)
        if process:
            is_alive = process.is_alive()
            log.info(f"Node {node_name} is {'running' if is_alive else 'not running'}.")
            return is_alive
        else:
            log.info(f"Node {node_name} is not being tracked.")
            return False

    def shutdown(self):
        log.info("Shutting down all nodes and ROS Master.")
        for node_name, process in self.node_processes.items():
            if process.is_alive():
                log.info(f"Stopping node {node_name}")
                process.stop()
        self.node_processes.clear()

        self.stop_launch_file()

        if self.launch:
            self.launch.shutdown()

        self.stop_roscore()
