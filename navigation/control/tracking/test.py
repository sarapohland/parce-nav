#!/usr/bin/env python3

import os
import sys
import cv2
import math
import torch
import rospy
import argparse
import threading
import ros_numpy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates

from navigation.control.planning.path_planning import PathPlanner
from navigation.control.tracking.path_tracking import PathTracker
from navigation.control.utils.coordinate_frames import world_to_robot, heading_from_quaternion


class Controller:

    ###########################
    ## Initialize controller ##
    ###########################

    def __init__(self, config_file, goal_x, goal_y):

        # Begin initializing controller
        self.initialized = False
        self.got_odom = False
        self.got_img = False
        self.init_position = [] 
        self.got_init_position = False
        self.reached_goal = False

        # Record selected and realized paths
        self.images = []
        self.init_states = []
        self.selected_paths = []
        self.realized_paths = []

        # Parse config file
        self.parse_xml(config_file)

        # Set initial state
        self.x = np.array([0.0, 0.0, 0.0])
        self.v = np.array([0.0, 0.0])

        # Intialize odometry subscriber
        if self.use_sim:
            self.odom_sub = rospy.Subscriber(self.odom_topic, ModelStates, self.odometry_callback, queue_size=1)
        else:
            self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odometry_callback, queue_size=1)
        
        # Listen for first odometry message
        if rospy.get_node_uri():
            while not self.got_odom:
                rospy.loginfo_throttle(10, f"Waiting for odom message on topic {self.odom_topic}")
                rospy.sleep(0.1)
        
        # Initialize image subscriber
        self.img_sub = rospy.Subscriber(self.img_topic, Image, self.image_callback, queue_size=1, buff_size=7372800)
       
        # Listen for first image
        if rospy.get_node_uri():
            while not self.got_img:
                rospy.loginfo_throttle(10, f"Waiting for first image on {self.img_topic}")
                rospy.sleep(0.1)

        # Set goal
        x, y, theta = self.init_position
        world_goal_x = x + float(goal_x) * np.cos(theta) - float(goal_y) * np.sin(theta)
        world_goal_y = y + float(goal_x) * np.sin(theta) + float(goal_y) * np.cos(theta)
        self.xG = np.array([world_goal_x, world_goal_y, 0.0])

        # Initialize velocity publisher
        self.vel_msg = Twist()
        self.vel_msg.linear.x = 0
        self.vel_msg.linear.y = 0
        self.vel_msg.linear.z = 0
        self.vel_msg.angular.x = 0
        self.vel_msg.angular.y = 0
        self.vel_msg.angular.z = 0
        self.vel_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        # Intialize path planner
        self.planning = PathPlanner(config_file) 

        # Initialize path-tracking controller
        self.tracking = PathTracker(config_file) 
        
        # Begin controller
        if rospy.get_node_uri():
            self.initialized = True 
            publisher = threading.Thread(target=self.publish_vel)
            publisher.start()
            self.control_loop()
        else:
            print('Warning: ROS node is not running!')


    ##################################
    ## Function to read config file ##
    ##################################

    def parse_xml(self, config_file):
        # Load XML config file
        tree = ET.parse(config_file)
        config = tree.getroot()

        # Read environment parameters
        environment = config.find("environment")
        self.vehicle_name = environment.find("vehicle").get("name")
        self.use_sim = environment.find("use_sim").get("enabled")
        self.use_sim = True if self.use_sim == "True" else False
        self.goal_thresh = float(environment.find("thresh").get("value"))
        
        # Read ROS topics
        topics = config.find("topics")
        self.odom_topic = topics.find("odom_topic").get("value")
        self.img_topic  = topics.find("img_topic").get("value")
        self.cmd_topic  = topics.find("cmd_topic").get("value")

        # Read control parameters
        controller = config.find("controller").find("control")
        self.u_low  = np.array([float(controller.find("u_low").get("lin")),
                                float(controller.find("u_low").get("ang"))])
        self.u_high = np.array([float(controller.find("u_high").get("lin")),
                                float(controller.find("u_high").get("ang"))])
        self.cmd_rate = float(controller.find("cmd_rate").get("value"))
        cmd_time = float(controller.find("cmd_time").get("value"))
        self.cmd_horizon = int(cmd_time * self.cmd_rate)

        # Read camera parameters
        camera = config.find("camera")
        self.image_size = [int(camera.find("size").get("channels")),
                           int(camera.find("size").get("height")),
                           int(camera.find("size").get("width"))]


    ###################################
    ## Subscriber callback functions ##
    ###################################

    # Get robot position
    def odometry_callback(self, msg):
        if self.use_sim:
            idx = msg.name.index(self.vehicle_name)
            pose = msg.pose[idx]
            twist = msg.twist[idx]
        else:
            pose = msg.pose.pose
            twist = msg.twist.twist

        # Set initial position
        if self.got_init_position == False: 
            yaw = heading_from_quaternion(pose.orientation)
            self.init_position = [pose.position.x, pose.position.y, yaw] 
            self.got_init_position = True
  
        # Set current position
        self.x[0] = pose.position.x
        self.x[1] = pose.position.y
        self.x[2] = heading_from_quaternion(pose.orientation)

        # Set current velocity
        self.v[0] = np.linalg.norm([twist.linear.x, twist.linear.y, twist.linear.z])
        self.v[1] = twist.angular.z

        # Get full state of vehicle
        self.state = np.concatenate((self.x, self.v))

        if self.initialized:
            # Check if vehicle reached goal
            dist_from_goal = np.linalg.norm(self.x[:2]-self.xG[:2])
            if (dist_from_goal < self.goal_thresh):
                self.reached_goal = True
                self.vel_msg.linear.x = 0
                self.vel_msg.angular.z = 0
                self.vel_pub.publish(self.vel_msg)
                self.realized_paths.append(self.realized_path)
                
                # End trial
                rospy.signal_shutdown("Finished trial!")
                sys.exit()

        self.got_odom = True

    # Get current RGB image
    def image_callback(self, msg):
        self.got_img = True
        cv_image = ros_numpy.numpify(msg)
        cv_image = np.swapaxes(np.swapaxes(cv_image, 1, 2), 0, 1)
        if (np.shape(cv_image)[1] != self.image_size[0] or np.shape(cv_image)[2] != self.image_size[1]):
            cv_image = cv2.resize(cv_image, dsize=(self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_CUBIC)
        self.image = torch.from_numpy(cv_image / 255).float()[None,:,:,:]

    ######################################
    ## Functions for running controller ##
    ######################################

    def control_loop(self):
        k = self.cmd_horizon
        self.realized_path = []
        rate = rospy.Rate(self.cmd_rate)
        while not rospy.is_shutdown():
            if self.reached_goal:
                self.vel_msg.linear.x = 0
                self.vel_msg.angular.z = 0

            else:
                # Record realized path
                self.realized_path.append(self.state)

                # If at end of planning horizon
                if k == self.cmd_horizon:
                    # Reset time step count
                    k = 0

                    # Save previously realized path
                    if len(self.realized_path) == (self.cmd_horizon + 1):
                        self.realized_paths.append(self.realized_path) 
                    self.realized_path = [self.state]

                    # Plan new path to goal
                    selected_path, ref_cmds = self.planning.plan_path(self.image, self.state, self.xG)

                    # Get state feedback matrices
                    K = self.tracking.compute_state_feedback(selected_path)

                    # xs = [pos[0] for pos in selected_path]
                    # ys = [pos[1] for pos in selected_path]
                    # plt.scatter(xs, ys, c='g', marker='.', s=10)
                    # plt.show()

                    # Save planned path from initial state
                    self.images.append(self.image)
                    self.init_states.append(np.copy(self.state))
                    self.selected_paths.append(selected_path)

                # Get optimal control
                u_ref = ref_cmds[k]
                x_ref = selected_path[k]
                u_opt = -K[k] @ (self.state - x_ref) + u_ref
                u = np.clip(u_opt, self.u_low, self.u_high)
                
                print("goal : ", self.xG)
                print("pos  : ", self.x)
                print('xref : ', x_ref)
                print('uref : ', u_ref)
                print("ctrl : ", u.flatten())
                
                # Publish control message
                self.vel_msg.linear.x = u[0]
                self.vel_msg.angular.z = u[1]

                # Increment time step
                k += 1
            rate.sleep()

    def publish_vel(self):
        rate = rospy.Rate(self.cmd_rate)
        while not rospy.is_shutdown():
            # Publish control message
            self.vel_pub.publish(self.vel_msg)
            rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('config_file', type=str)
    parser.add_argument('--goal_x', type=float, default=20.0)
    parser.add_argument('--goal_y', type=float, default=0.0)
    args = parser.parse_args()

    rospy.init_node('controller', anonymous=True, disable_signals=True)
    interface = Controller(args.config_file, args.goal_x, args.goal_y)

    # Create folder to save results
    output_dir = 'results/tracking/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot sampled paths and realized paths
    for idx, (img, init_state, desired, realized) in enumerate(zip(interface.images, interface.init_states, interface.selected_paths, interface.realized_paths)):

        # Compute error in position
        desired = desired[:len(realized)]
        error = realized - desired
        pos_error = np.linalg.norm(error[:,:2], axis=1)

        # Display image
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplot(1, 2, 1)
        img = np.squeeze(img.numpy() * 255).astype(np.uint8)
        img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
        plt.imshow(img)

        # Display selected trajectory
        new_desired = world_to_robot(np.array(desired), init_state[:3])
        new_desired = interface.planning.get_traj_pixels(new_desired)
        xs = [pos[0] for pos in new_desired]
        ys = [pos[1] for pos in new_desired]
        plt.scatter(xs, ys, c='g', marker='.', s=10, label='Desired')

        # Display realizd trajectory
        new_realized = world_to_robot(np.array(realized), init_state[:3])
        new_realized = interface.planning.get_traj_pixels(new_realized)
        xs = [pos[0] for pos in new_realized]
        ys = [pos[1] for pos in new_realized]
        plt.scatter(xs, ys, c='b', marker='.', s=10, label='Realized')

        plt.legend()
        plt.title('Panned Path and Realized Trajectory')

        # Display position error over time
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(pos_error)), pos_error, c='r', marker='.', s=5)
        plt.title('Error in Vehicle Position')
        plt.xlabel('Time Step')
        plt.ylabel('Position Error (m)')

        plt.suptitle('Performance of Path-Tracking Controller')
        plt.savefig(os.path.join(output_dir, '({},{})_{}.png'.format(round(interface.xG[0]), round(interface.xG[1]), idx)))
        plt.close()
