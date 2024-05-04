#!/usr/bin/env python3

import os
import sys
import csv
import cv2
import math
import torch
import rospy
import random
import pickle
import argparse
import threading
import ros_numpy
import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ContactsState

from navigation.control.planning.path_planning import PathPlanner
from navigation.control.tracking.path_tracking import PathTracker
from navigation.control.utils.coordinate_frames import world_to_robot, heading_from_quaternion
from navigation.control.planning.test import plot_trajs


class Controller:

    ###########################
    ## Initialize controller ##
    ###########################

    def __init__(self, config_file=None, debug=False, record=False, visualize=False):
        self.debug = debug
        self.record = record
        self.visualize = visualize

        # Begin initializing controller
        self.initialized = False
        self.got_odom = False
        self.got_img = False
        self.init_position = []
        self.got_init_position = False

        # Record vehicle data
        self.time = []
        self.vel_lin = []
        self.vel_ang = []
        self.cmd_lin = []
        self.cmd_ang = []
        self.vel = [0,0,0,0,0,0]
        self.state = [0,0,0,0,0]
        self.path_length = 0
        self.collision = False
        self.has_collided = False

        # Parse config file
        if config_file is None:
            config_file = rospy.get_param("~config")
        self.parse_xml(config_file)

        # Create dictionary to store data for visualization
        if self.visualize:
            self.vis_data = {'config': config_file, 'time': [], 'state': [], 
                                'image': [], 'goal': [], 'trajs': [],
                                'comp': [], 'regnl': [], 'cmap': []}

        # Listen for simulation time
        self.sim_time = None
        if self.use_sim:
            self.clock_sub = rospy.Subscriber(self.clk_topic, Clock, self.clock_callback, queue_size=1)

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
        
        # Initialize goals
        self.goal_counter = 0
        self.goal_reached = False
        if self.randGoal:
            self.set_rand_goal()
        elif self.global_path:
            self.create_goal_list()
        elif self.local_path:
            self.create_goal_list(local=True)

        # Listen for vehicle collision
        self.collision_sub = rospy.Subscriber(self.coll_topic, ContactsState, self.collision_callback, queue_size=1)

        # Initialize velocity publisher
        self.vel_msg = Twist()
        self.vel_msg.linear.x = 0
        self.vel_msg.linear.y = 0
        self.vel_msg.linear.z = 0
        self.vel_msg.angular.x = 0
        self.vel_msg.angular.y = 0
        self.vel_msg.angular.z = 0
        self.vel_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        # Intialize path planner and tracking controller
        self.planning = PathPlanner(config_file) 
        self.tracking = PathTracker(config_file)
        self.K = self.ref_cmds = self.selected_path = None
        planner = threading.Thread(target=self.plan_path)
        planner.start()
        
        # Wait for initial path
        if rospy.get_node_uri():
            while self.K is None:
                rospy.loginfo_throttle(10, "Waiting for planner to generate initial path.")
                rospy.sleep(0.1)
        
        # Begin controller
        if rospy.get_node_uri():
            self.initialized = True 
            self.timer_start = self.sim_time
            self.fsm = 'turn'
            self.ctrl_loop()
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
        self.time_limit = float(environment.find("time_limit").get("value"))
        self.goal_theta = float(environment.find("max_theta").get("value"))
        
        self.x = np.array([float(environment.find("xI").get("x")),
                           float(environment.find("xI").get("y")),
                           float(environment.find("xI").get("th"))])
        self.randGoal = environment.find("randGoal").get("enabled")
        self.global_path = environment.find("global_waypoints").get("enabled")
        self.local_path = environment.find("local_waypoints").get("enabled")

        # Parse goal information
        if self.randGoal == "True":
            self.randGoal = True
            self.global_path = self.local_path = False
            xmin = float(environment.find("randGoal").find("min").get("x"))
            ymin = float(environment.find("randGoal").find("min").get("y"))
            xmax = float(environment.find("randGoal").find("max").get("x"))
            ymax =float(environment.find("randGoal").find("max").get("y"))
            self.min_dist = float(environment.find("randGoal").find("min_dist").get("value"))
            self.max_dist = float(environment.find("randGoal").find("max_dist").get("value"))
            self.max_theta = float(environment.find("randGoal").find("max_theta").get("value"))
            self.x_range = [xmin, xmax]
            self.y_range = [ymin, ymax]
        
        elif self.global_path == "True":
            self.global_path = True
            self.randGoal = self.local_path = False
            self.global_path_csv = environment.find("global_waypoints").find("waypoint_file").get("value")
        
        elif self.local_path == "True":
            self.local_path = True
            self.randGoal = self.global_path = False
            self.local_path_csv = environment.find("local_waypoints").find("waypoint_file").get("value")
       
        else:
            self.xG = np.array([float(environment.find("xG").get("x")),
                            float(environment.find("xG").get("y")),
                            float(environment.find("xG").get("th"))])

        # Read ROS topics
        topics = config.find("topics")
        self.odom_topic = topics.find("odom_topic").get("value")
        self.img_topic  = topics.find("img_topic").get("value")
        self.cmd_topic  = topics.find("cmd_topic").get("value")
        self.clk_topic  = topics.find("clk_topic").get("value")
        self.coll_topic = topics.find("coll_topic").get("value")

        # Read control parameters
        controller = config.find("controller").find("control")
        self.u_low  = np.array([float(controller.find("u_low").get("lin")),
                                float(controller.find("u_low").get("ang"))])
        self.u_high = np.array([float(controller.find("u_high").get("lin")),
                                float(controller.find("u_high").get("ang"))])
        self.cmd_rate    = float(controller.find("cmd_rate").get("value"))
        self.cmd_time    = float(controller.find("cmd_time").get("value"))
        self.backup_time = float(controller.find("backup_time").get("value"))
        self.turn_time   = float(controller.find("turn_time").get("value"))
        self.error_time  = float(controller.find("error_time").get("value"))

        # Read camera parameters
        camera = config.find("camera")
        self.image_size = [int(camera.find("size").get("channels")),
                           int(camera.find("size").get("height")),
                           int(camera.find("size").get("width"))]


    ##############################################
    ## Functions for setting and updating goals ##
    ##############################################

    def set_rand_goal(self):
        iter_flag = False 
        iter_count = 0 
        max_iter = 10000

        if len(self.init_position) < 3:
            self.init_position = [0, 0, np.pi/2]

        while not iter_flag: 
            # Generate goal in bounds
            rand_goal = [
                np.random.uniform(low=self.x_range[0], high=self.x_range[1]),
                np.random.uniform(low=self.y_range[0], high=self.y_range[1]),
                0,
            ]
            self.xG = [
                self.init_position[0] + float(rand_goal[0])*math.cos(self.init_position[2]),
                self.init_position[1] + float(rand_goal[1])*math.sin(self.init_position[2]),
                0,
            ]
            vec_to_goal = np.array(self.xG[:2]) - np.array(self.x[:2])

            # Check distance from current position to goal
            dist = np.linalg.norm(vec_to_goal)

            # Check angle between current position and goal3
            theta = np.arctan2(vec_to_goal[1], vec_to_goal[0])
            theta = theta - self.x[2]
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            # Check if randomly generated goal is valid
            if dist < self.max_dist and dist > self.min_dist and np.abs(theta) < self.max_theta:
                iter_flag = True 
            else: 
                iter_count += 1 
                if iter_count > max_iter: 
                    self.xG = [
                        self.init_position[0],
                        self.init_position[1],
                        0,
                    ] 
                    print("Max goal iteration reached, returning to initial position")

        print('Random goal set: ', self.xG)
    
    def create_goal_list(self, local=False):

        self.goal_list = []
        if len(self.init_position) < 3:
            self.init_position = [0, 0, np.pi/2]

        if local:
            print('Local goal CSV file: ', self.local_path_csv)
            with open(self.local_path_csv) as csvfile:
                goal_cnt = 0
                for row in csv.reader(csvfile, delimiter=','):
                    print("csv row = ", row)
                    goal_x = float(self.init_position[0]) + float(row[0]) * math.cos(self.init_position[2])
                    goal_y = float(self.init_position[1]) + float(row[1]) * math.sin(self.init_position[2])  
                    goal_theta = float(row[2]) + (self.init_position[2])
                    goal_theta = (goal_theta + np.pi) % (2 * np.pi) - np.pi
                    goal = np.array([goal_x, goal_y, goal_theta])
                    self.goal_list.append(goal)
                    goal_cnt +=1

        else:
            print('Global goal CSV file: ', self.global_path_csv)
            with open(self.global_path_csv) as csvfile:
                goal_cnt = 0
                for row in csv.reader(csvfile, delimiter=','):
                    goal_x = float(row[0])
                    goal_y = float(row[1])  
                    goal_theta = float(row[2])
                    goal_theta = (goal_theta + np.pi) % (2 * np.pi) - np.pi
                    goal = np.array([goal_x, goal_y, goal_theta])
                    self.goal_list.append(goal)
                    goal_cnt +=1

        self.xG = self.goal_list[0]
        print('Goal list set: ', self.xG)
    
    def set_waypoint(self):
        l = len(self.goal_list)
        for i in range(0, l):
            if(np.array_equal(self.xG, self.goal_list[i], equal_nan=True)):
                if i != (l-1):
                    self.xG = self.goal_list[i+1]
                    print('Next waypoint set: ', self.xG)
                    # if(i + 2) == l:
                    #     self.goal_thresh = 1.0
                    break
                else:
                    self.goal_reached = True
                    self.record_results('success')
        self.fsm = 'turn'


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
            print("Initial position: ", self.init_position)
  
        # Set current position
        prev_pos = self.x.copy()
        self.x[0] = pose.position.x
        self.x[1] = pose.position.y
        self.x[2] = heading_from_quaternion(pose.orientation)

        # Set current velocity
        self.vel[0] = twist.linear.x
        self.vel[1] = twist.linear.y
        self.vel[2] = twist.linear.z
        self.vel[3] = twist.angular.x
        self.vel[4] = twist.angular.y
        self.vel[5] = twist.angular.z

        # Record vehicle velocity
        self.time.append(self.sim_time.to_sec())
        self.vel_lin.append(np.linalg.norm(self.vel[0:3]))
        self.vel_ang.append(self.vel[5])

        # Get full state of vehicle
        self.state[0:3] = self.x
        self.state[3] = np.linalg.norm(self.vel[0:3])
        self.state[4] = self.vel[5]

        # Update path length
        self.path_length += np.linalg.norm(prev_pos[0:2] - self.x[0:2])

        if self.initialized:
            # Get navigation time and distance to goal
            self.nav_time = (self.sim_time - self.timer_start).to_sec()
            self.dist_from_goal = np.linalg.norm(self.x[:2]-self.xG[:2])

            # Check if vehicle reached the time limit
            if self.time_limit is not None and (self.time_limit > 0 and self.nav_time > self.time_limit):
                self.vel_msg.linear.x = 0
                self.vel_msg.angular.z = 0
                self.vel_pub.publish(self.vel_msg)
                self.record_results('timeout')

            # Check if vehicle reached goal
            if (self.dist_from_goal < self.goal_thresh):
                self.vel_msg.linear.x = 0
                self.vel_msg.angular.z = 0
                self.vel_pub.publish(self.vel_msg)
                self.goal_counter += 1 

                if self.randGoal:
                    self.set_rand_goal()
                elif self.global_path:
                    self.set_waypoint()
                elif self.local_path:
                    self.set_waypoint()

            # Check if vehicle has rolled over
            quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            roll = R.from_quat(quaternion).as_euler('zyx')[2]
            if roll > np.pi / 2:
                self.vel_msg.linear.x = 0
                self.vel_msg.angular.z = 0
                self.vel_pub.publish(self.vel_msg)
                self.record_results('rollover')
        
        self.got_odom = True

    # Get current RGB image
    def image_callback(self, msg):
        self.got_img = True
        cv_image = ros_numpy.numpify(msg)
        cv_image = np.swapaxes(np.swapaxes(cv_image, 1, 2), 0, 1)
        if (np.shape(cv_image)[1] != self.image_size[0] or np.shape(cv_image)[2] != self.image_size[1]):
            cv_image = cv2.resize(cv_image, dsize=(self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_CUBIC)
        self.image = torch.from_numpy(cv_image / 255).float()[None,:,:,:]

    # Get current simulation time
    def clock_callback(self, msg):
        self.sim_time = msg.clock

    # Check whether vehicle collision occurred
    def collision_callback(self, msg):
        if len(msg.states) > 0:
            self.collision = True
            self.has_collided = True
        else:
            self.collision = False


    ######################################
    ## Functions for running controller ##
    ######################################

    def _turn_direction(self):
        vec_to_goal = np.array(self.xG[:2]) - np.array(self.x[:2])
        theta = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        theta = theta - self.x[2]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        if theta > self.goal_theta:
            return 'left'
        elif theta < -self.goal_theta:
            return 'right'
        else:
            return None
    
    # TODO: Incorporate robot stuck response into control loop
    def _is_robot_stuck(self):
        # Grab velocities and commands from last several seconds
        num_error = int(self.error_time * self.cmd_rate)
        lin_vels = np.abs(self.vel_lin[-num_error:])
        ang_vels = np.abs(self.vel_ang[-num_error:])
        lin_cmds = np.abs(self.cmd_lin[-num_error:])
        ang_cmds = np.abs(self.cmd_ang[-num_error:])

        # Check if vehicle is stuck
        # TODO: Check accurcay of robot stuck function
        if np.all(lin_vels < 0.1) and np.all(ang_vels < 0.1):
            if not np.all(lin_cmds < 0.1) and not np.all(ang_cmds < 0.1):
                return True
        return False
    
    def plan_path(self):
        rate = rospy.Rate(self.cmd_rate)
        visualize = self.debug or self.visualize
        while not rospy.is_shutdown():
            # Plan new path to goal
            self.img_of_path, self.goal_of_path = self.image, self.xG
            self.selected_path, self.ref_cmds = self.planning.plan_path(self.image, self.state, self.xG, visualize)
            if visualize:
                self.sampled_paths = self.planning._sampled_paths
                self.overall = self.planning._competency
                self.regional = self.planning._regional
                self.comp_map = self.planning._comp_map

            # If there is no good path, switch to safe response
            if self.selected_path is None:
                self.K = 0
                self.safe_response = True
            
            # Otherwise, get state feedback matrices
            else:
                self.K = self.tracking.compute_state_feedback(self.selected_path)
                self.safe_response = False
            
            rate.sleep()

    def ctrl_loop(self):
        ref_cmds = None
        rate = rospy.Rate(self.cmd_rate)
        while not rospy.is_shutdown():
            # Stop vehicle once final goal is reached
            if self.goal_reached:
                print('Vehicle has reached goal, stopping.')
                self.fsm = 'stop'

            if self.fsm == 'stop':
                self.vel_msg.linear.x = 0
                self.vel_msg.angular.z = 0

            # Turn towards goal after receiving new goal
            if self.fsm == 'turn':
                turn_direction = self._turn_direction()
                if turn_direction == 'left':
                    print('Turning {} towards next goal.'.format(turn_direction))
                    self.vel_msg.linear.x = 0
                    self.vel_msg.angular.z = self.u_high[1]
                elif turn_direction == 'right':
                    print('Turning {} towards next goal.'.format(turn_direction))
                    self.vel_msg.linear.x = 0
                    self.vel_msg.angular.z = self.u_low[1]
                else:
                    self.fsm = 'plan'
                    if not np.all(self.goal_of_path == self.xG):
                        print('Planning path towards new goal.')
                        continue

            # Grab planned path to goal at start of new control sequence
            image = goal = trajs = comp = regnl = cmap = None
            if self.fsm == 'plan':
                print('Planning new path towards goal.')
                k = 0
                K = self.K
                ref_cmds = self.ref_cmds
                selected_path = self.selected_path
                # self.fsm = 'lqr'
                self.fsm = 'safe' if self.safe_response else 'lqr'

                if self.debug:
                    plot_trajs(self.img_of_path, self.planning)
                
                if self.visualize:
                    image = self.img_of_path
                    goal  = self.goal_of_path
                    trajs = self.sampled_paths
                    comp  = self.overall
                    regnl = self.regional
                    cmap  = self.comp_map

            # # Switch to safe response if there is no safe path
            # if self.fsm == 'lqr' and self.safe_response:
            #     k = 0
            #     ref_cmds = self.ref_cmds
            #     self.fsm = 'safe'
            
            if self.fsm == 'safe':
                print('Executing action in safe response.')
                u = ref_cmds[k]
                self.vel_msg.linear.x = u[0]
                self.vel_msg.angular.z = u[1]
                k += 1

                print('xG: ', np.round(self.xG, 2))
                print('x:  ', np.round(self.state, 2))
                print('u:  ', np.round(u, 2))
            
            # Determine optimal control along safe path using LQR
            if self.fsm == 'lqr':
                print('Determining optimal control using LQR.')
                u_ref = ref_cmds[k]
                x_ref = selected_path[k]
                u_opt = -K[k] @ (self.state - x_ref) + u_ref
                u = np.clip(u_opt, self.u_low, self.u_high)
                self.vel_msg.linear.x = u[0]
                self.vel_msg.angular.z = u[1]
                k += 1
            
                print('xG:   ', np.round(self.xG, 2))
                print('xref: ', np.round(x_ref, 2))
                print('x:    ', np.round(self.state, 2))
                print('uref: ', np.round(u_ref, 2))
                print('u:    ', np.round(u, 2))

            # Reached the end of control sequence; need to replan
            if ref_cmds is not None and k == len(ref_cmds):
                self.fsm = 'plan'

            # Save visualization data
            if self.visualize:
                self.vis_data['time'].append(self.sim_time.to_sec())
                self.vis_data['state'].append(np.copy(self.state))
                self.vis_data['image'].append(image)
                self.vis_data['comp'].append(comp)
                self.vis_data['regnl'].append(regnl)
                self.vis_data['cmap'].append(cmap)
                self.vis_data['goal'].append(goal)
                self.vis_data['trajs'].append(trajs)

            # Record and publish velocity command
            self.cmd_lin.append(self.vel_msg.linear.x)
            self.cmd_ang.append(self.vel_msg.angular.z)
            self.vel_pub.publish(self.vel_msg)
            rate.sleep()


    #####################################
    ## Functions for recording results ##
    #####################################

    def check_end_collision(self):
        collisions = []
        start = self.sim_time.to_sec()
        while (self.sim_time.to_sec() - start) < 1:
            collisions.append(self.collision)
        end_collision = np.sum(np.array(collisions)) > len(collisions)/3
        return end_collision

    def record_results(self, result):
        # Dump visualization data
        if self.visualize:
            folder = os.path.dirname(self.visualize)
            if not os.path.exists(folder):
                os.makedirs(folder)
            pickle.dump(self.vis_data, open(self.visualize, 'wb'))

        # Check end result of trial
        if result == 'success':
            end_result = 'success'
            print('Reached final goal!')
        elif result == 'rollover':
            end_result = 'rollover'
            self.nav_time = self.time_limit
            print('Vehicle has rolled over.')
        elif self.check_end_collision():
            end_result = 'collision'
            print('Ending on collision.')
        else:
            end_result = 'timeout'
            print('Time limit reached.')

        # Compute distance to final goal
        final_goal_dist = np.linalg.norm(self.x[:2]-self.goal_list[-1][:2])

        # Compute average velocity and acceleration
        time = np.array(self.time)
        vel_lin = np.array(self.vel_lin)
        vel_ang = np.array(self.vel_ang)
        acc_lin = np.diff(vel_lin) / np.diff(time)
        acc_ang = np.diff(vel_ang) / np.diff(time)

        vel_lin_avg = np.mean(np.abs(vel_lin))
        vel_ang_avg = np.mean(np.abs(vel_ang))
        acc_lin_avg = np.mean(np.abs(np.ma.masked_invalid(acc_lin)))
        acc_ang_avg = np.mean(np.abs(np.ma.masked_invalid(acc_ang)))
        
        # Print performance metrics
        print('Number of goals reached: ', self.goal_counter)
        print("Distance from final goal: ", final_goal_dist)
        print("Total navigation time: ", self.nav_time)
        print("Total path length: ", self.path_length)
        print("Did collision occur?: ", self.has_collided)
        print("Average linear speed: ", vel_lin_avg)
        print("Average angular speed: ", vel_ang_avg)
        print("Average linear acceleration: ", acc_lin_avg)
        print("Average angular acceleration: ", acc_ang_avg)

        # Record performance metrics
        if self.record:
            file = self.record
            folder = os.path.dirname(self.record)
            if not os.path.exists(folder):
                os.makedirs(folder)
            if os.path.isfile(file):
                df = pd.read_csv(file)
                df.loc[df.index.max() + 1] = [self.planning.use_ovl_comp, 
                                              self.planning.use_reg_comp,
                                              self.planning.use_traj_resp,
                                              self.planning.ovl_thresh,
                                              self.planning.reg_thresh,
                                              Path(self.global_path_csv).stem,
                                              end_result,
                                              self.goal_counter,
                                              final_goal_dist,
                                              self.nav_time,
                                              self.path_length,
                                              self.has_collided,
                                              vel_lin_avg,
                                              vel_ang_avg,
                                              acc_lin_avg,
                                              acc_ang_avg]
            else:
                trial = {'Overall': [self.planning.use_ovl_comp], 
                         'Regional': [self.planning.use_reg_comp], 
                         'Trajectory': [self.planning.use_traj_resp], 
                         'Overall Bound': [self.planning.ovl_thresh], 
                         'Regional Bound': [self.planning.reg_thresh], 
                         'Scenario': [Path(self.global_path_csv).stem], 
                         'Result': [end_result], 
                         'Wypts Reached': [self.goal_counter], 
                         'Dist to Goal': [final_goal_dist], 
                         'Nav Time': [self.nav_time], 
                         'Path Length': [self.path_length], 
                         'Collision': [self.has_collided],
                         'Lin Vel': [vel_lin_avg],
                         'Ang Vel': [vel_ang_avg],
                         'Lin Accel': [acc_lin_avg],
                         'Ang Accel': [acc_ang_avg]}
                df = pd.DataFrame.from_dict(trial)
            df.to_csv(file, index=False)
            print("Recorded results to ", file)

        # End trial
        rospy.signal_shutdown("Finished trial!")
        sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('config_file', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--record', type=str, default=None)
    parser.add_argument('--visualize', type=str, default=None)
    args = parser.parse_args()

    rospy.init_node('controller', anonymous=True, disable_signals=True)
    interface = Controller(args.config_file, args.debug, args.record, args.visualize)
    rospy.spin()