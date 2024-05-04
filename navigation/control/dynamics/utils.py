import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from bagpy import bagreader
from ast import literal_eval
from scipy.signal import butter, lfilter

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Vector3
from geometry_msgs.msg import Pose, Point, Quaternion

from navigation.control.utils.coordinate_frames import heading_from_quaternion

V_MIN, V_MAX = -1.0, 1.0
W_MIN, W_MAX = -np.pi/2, np.pi/2

def lowpass_filter(arr):
    order, cutoff, fs = 4, 1, 10
    b, a = butter(order, cutoff, fs=fs, btype='lowpass', analog=False)
    return lfilter(b, a, arr)

def remove_outliers(arr, std):
    zscores = np.abs(stats.zscore(arr))
    for i1 in range(len(arr)):
        i2 = i1
        while np.abs(zscores[i2]) > std and i2 > 0:
            i2 -= 1
        arr[i1] = arr[i2]
    return arr

def rolling_avg(arr, window):
    new_arr = np.zeros(len(arr))
    for i in range(len(arr)):
        start = np.maximum(0, i-window)
        new_arr[i] = np.mean(arr[start:start+window])
    return new_arr

def list_files(path):
    files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(os.path.join(path, file))
    return files

def parse_pose_msg(pose_str, idx=None):
    # Get position and orientation strings from Pose string
    if idx is None:
        pos_str = re.findall(r'position:(.*?)orientation:', pose_str, re.DOTALL)
        ori_str = re.findall(r'orientation:(.*?)(?=position:|\])', pose_str, re.DOTALL)
    else:
        pos_str = re.findall(r'position:(.*?)orientation:', pose_str, re.DOTALL)[idx]
        ori_str = re.findall(r'orientation:(.*?)(?=position:|\])', pose_str, re.DOTALL)[idx]

    # Create list of positions and orientations
    pos = re.findall(r'[-+]?\d*\.\d+|\d+', pos_str)
    ori = re.findall(r'[-+]?\d*\.\d+|\d+', ori_str)

    # Get Pose message from data
    pose_msg = Pose()
    pose_msg.position = Point(float(pos[0]), float(pos[1]), float(pos[2]))
    pose_msg.orientation = Quaternion(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]))
    return pose_msg

def parse_twist_msg(twist_str, idx=None):
    # Get linear and angular velocity strings from Twist string
    if idx is None:
        lin_str = re.findall(r'linear:(.*?)angular:', twist_str, re.DOTALL)
        ang_str = re.findall(r'angular:(.*?)(?=linear:|\])', twist_str, re.DOTALL)
    else:
        lin_str = re.findall(r'linear:(.*?)angular:', twist_str, re.DOTALL)[idx]
        ang_str = re.findall(r'angular:(.*?)(?=linear:|\])', twist_str, re.DOTALL)[idx]

    # Create list of linear and angular velocities
    lin = re.findall(r'x: ([-+]?\d*\.\d+|\d+|-?\d+\.\d+e[+-]?\d+|-?\d+e[+-]?\d+)\s+y: ([-+]?\d*\.\d+|\d+|-?\d+\.\d+e[+-]?\d+|-?\d+e[+-]?\d+)\s+z: ([-+]?\d*\.\d+|\d+|-?\d+\.\d+e[+-]?\d+|-?\d+e[+-]?\d+)', lin_str)
    ang = re.findall(r'x: ([-+]?\d*\.\d+|\d+|-?\d+\.\d+e[+-]?\d+|-?\d+e[+-]?\d+)\s+y: ([-+]?\d*\.\d+|\d+|-?\d+\.\d+e[+-]?\d+|-?\d+e[+-]?\d+)\s+z: ([-+]?\d*\.\d+|\d+|-?\d+\.\d+e[+-]?\d+|-?\d+e[+-]?\d+)', ang_str)
        
    # Get Twist message from data
    twist_msg = Twist()
    twist_msg.linear = Vector3(float(lin[0][0]), float(lin[0][1]), float(lin[0][2]))
    twist_msg.angular = Vector3(float(ang[0][0]), float(ang[0][1]), float(ang[0][2]))
    return twist_msg

def extract_bags(files, cmd_topic, odom_topic, vehicle_name=None, visualize=False):
    times, states, inputs = [], [], []
    # Create odometry and command data frames
    for file in files:
        # Read bag file
        b = bagreader(file)

        # Create csv files of relevant topics from bag file
        cmds  = b.message_by_topic(cmd_topic)
        odom  = b.message_by_topic(odom_topic)

        # Create data frames from csv files
        cmds_df = pd.read_csv(cmds)
        odom_df = pd.read_csv(odom)

        # Get velocity commands
        cmd_t  = np.asarray(cmds_df['Time'])
        cmd_vt = np.asarray(cmds_df["linear.x"])
        cmd_vs = np.asarray(cmds_df["angular.z"])

        # Get vehicle odometry from ModelStates message
        if vehicle_name is not None:
            # import pdb; pdb.set_trace()
            odom_t  = np.asarray(odom_df['Time'])

            # Get vehicle index
            all_names = odom_df['name'].apply(literal_eval).iloc[0]
            idx = all_names.index(vehicle_name)

            # Parse data to compile position information
            odom_x, odom_y, odom_ori = [], [], []
            for pose_str in odom_df['pose']: 
                pose_msg = parse_pose_msg(pose_str, idx)
                odom_x.append(pose_msg.position.x)
                odom_y.append(pose_msg.position.y)
                odom_ori.append(heading_from_quaternion(pose_msg.orientation))
            
            # Parse data to compile velocity information
            odom_vel, odom_ang = [], []
            for twist_str in odom_df['twist']:
                twist_msg = parse_twist_msg(twist_str, idx)
                odom_vel.append(np.linalg.norm([twist_msg.linear.x, twist_msg.linear.y]))
                odom_ang.append(twist_msg.angular.z)
            
            # Convert lists to numpy arrays
            odom_x   = np.array(odom_x)
            odom_y   = np.array(odom_y)
            odom_ori = np.array(odom_ori)
            odom_vel = np.array(odom_vel)
            odom_ang = np.array(odom_ang)

            # Clip measurements to valid ranges
            odom_vel = np.clip(odom_vel, V_MIN, V_MAX)
            odom_ang = np.clip(odom_ang, W_MIN, W_MAX)

        # Get vehicle odometry from Odometry message
        else:
            # Parse data to compile position information
            from scipy.spatial.transform import Rotation as R
            odom_t  = np.asarray(odom_df['Time'])
            odom_x = np.asarray(odom_df["pose.pose.position.x"])
            odom_y = np.asarray(odom_df["pose.pose.position.y"])
            quaternion = np.array(["pose.pose.orientation.x",
                                    "pose.pose.orientation.y",
                                    "pose.pose.orientation.z",
                                    "pose.pose.orientation.w"])
            odom_ori = R.from_quat(quaternion).as_euler('zyx')[0]

            # Parse data to compile velocity information
            odom_vx = np.asarray(odom_df["twist.twist.linear.x"])
            odom_vy = np.asarray(odom_df["twist.twist.linear.y"])
            odom_vel = np.linalg.norm(np.stack((odom_vx, odom_vy)), axis=0)
            odom_ang = np.asarray(odom_df["twist.twist.angular.z"])

        # Remove outliers from noisy measurments
        odom_x   = remove_outliers(np.copy(odom_x), 3)
        odom_y   = remove_outliers(np.copy(odom_y), 3)
        odom_ori = remove_outliers(np.copy(odom_ori), 3)
        odom_vel = remove_outliers(np.copy(odom_vel), 3)
        odom_ang = remove_outliers(np.copy(odom_ang), 3)

        # Interpolate data at input times
        odom_x   = np.interp(cmd_t, odom_t,  odom_x)
        odom_y   = np.interp(cmd_t, odom_t,  odom_y)
        odom_ori = np.interp(cmd_t, odom_t,  odom_ori)
        odom_vel = np.interp(cmd_t, odom_t,  odom_vel)
        odom_ang = np.interp(cmd_t, odom_t,  odom_ang)

        # Compile preprocessed data
        times.append(cmd_t - cmd_t[0])
        inputs.append(np.vstack([cmd_vt, cmd_vs]))
        states.append(np.vstack([odom_x, odom_y, odom_ori, odom_vel, odom_ang]))

    # Compile all preprocessed data
    T = np.hstack(times)
    U = np.hstack(inputs)
    X = np.hstack(states)
  
    # Display all preprocessed data
    if visualize:
        import matplotlib.pyplot as plt
        plt.title('X Position')
        plt.scatter(range(np.shape(X)[1]), X[0,:])
        plt.show()
        plt.title('Y Position')
        plt.scatter(range(np.shape(X)[1]), X[1,:])
        plt.show()
        plt.title('Orientation')
        plt.scatter(range(np.shape(X)[1]), X[2,:])
        plt.show()
        plt.title('Linear Velocity')
        plt.scatter(range(np.shape(X)[1]), X[3,:])
        plt.show()
        plt.title('Turn Rate')
        plt.scatter(range(np.shape(X)[1]), X[4,:])
        plt.show()

    return T, U, X
