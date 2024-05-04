import rospy
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from geometry_msgs.msg import Twist

class RandomWalk:

    def __init__(self, config_file=None):
        # Parse config file
        if config_file is None:
            config_file = rospy.get_param("~config")
        self.parse_xml(config_file)

        # Initialize comman publisher
        self.vel_msg = Twist()
        self.vel_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        # Set random walk parameters
        lin_accel = 0.1  # unbiased maximum linear acceleration (m/s^2)
        ang_accel = 0.15 # unbiased maximum angular acceleration (rad/s^2)
        lin_bias  = 0.05 # bias to offset linear acceleration range
        ang_bias  = 0.075 # bias to offset angular acceleration range

        # Initialize command sequence
        steps = int(self.T * self.cmd_rate) # number of time steps
        cmds = np.zeros((2, steps))

        # Sample sequence of linear and angular velocity commands
        lin_range = np.array([-lin_accel, lin_accel])
        ang_range = np.array([-ang_accel, ang_accel])
        switch = ((np.array(self.u_high) - np.array(self.u_low)) / np.array([lin_accel, ang_accel])).astype(int)
        for i, prev in enumerate(cmds.T):
            if i == np.shape(cmds)[1]-1:
                break
            if i % switch[0] == 0:
                lin_bias *= -1
                lin_range = (np.array([-lin_accel, lin_accel]) + lin_bias) * self.cmd_rate
            if i % switch[1] == 0:
                ang_bias *= -1
                ang_range = (np.array([-ang_accel, ang_accel]) + ang_bias) * self.cmd_rate
            delta = np.random.uniform([lin_range[0], ang_range[0]], [lin_range[1], ang_range[1]])
            cmds[:,i+1] = np.clip(prev + delta, self.u_low, self.u_high)

        # Begin running random walk
        self.random_walk(cmds)

    def parse_xml(self, config_file):
        # Load XML config file
        tree = ET.parse(args.config_file)
        config = tree.getroot()
        
        # Read environment parameters
        environment = config.find("environment")
        self.use_sim = environment.find("use_sim").get("enabled")
        self.use_sim = True if self.use_sim == "True" else False
        self.T = float(environment.find("time_limit").get("value"))

        # Read ROS topics
        topics = config.find("topics")
        self.cmd_topic = topics.find("cmd_topic").get("value")
        
        # Read controller parameters
        control = config.find("controller").find("control")
        self.cmd_rate = float(control.find("cmd_rate").get("value"))
        self.u_low = np.array([float(control.find("u_low").get("lin")),
                                float(control.find("u_low").get("ang"))])
        self.u_high = np.array([float(control.find("u_high").get("lin")),
                                float(control.find("u_high").get("ang"))])

    def random_walk(self, cmds):
        step = 0
        rate = rospy.Rate(self.cmd_rate)
        self.start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown() and (rospy.Time.now().to_sec() - self.start_time < self.T):
            # Publish current command
            cmd = cmds[:,step]
            self.vel_msg.linear.x  = cmd[0]
            self.vel_msg.angular.z = cmd[1]
            self.vel_pub.publish(self.vel_msg)
            step += 1
            rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()
    
    rospy.init_node('random_walk', anonymous=True, disable_signals=True)
    interface = RandomWalk(args.config_file)
    rospy.spin()
