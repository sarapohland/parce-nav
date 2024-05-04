# Navigation Control Configurations

## Environment

The first section of this XML file defines the environment parameters. The first two parameters define the default initial position and goal position if random goals and waypoints are not used. The next 7 lines allow you to determine if and how random goals are sampled, and the following 6 lines allow you to specify a CSV file with waypoints given in global or local coordinates. The parameter thresh determines how close the vehicle must get to a goal/waypoint before moving on to the next one, use_sim indicates whether trials are being run in simulation or on a physical platform, vehicle_name specifies which vehicle model should be used (husky, spot, or warty), time_limit indicates the amount of time allocated for the given trial, and max_theta determines the angle at which a goal is considered to be behind the vehicle. (The vehicle will orient itself towards the goal upon receiving a new goal.)

## Topics

The second section of the XML file allows you to define relevant ROS topics. In this section, odom_topic is the topic the controller will subscribe to for vehicle odometry information, img_topic should contain the front-facing camera images, cmd_topic is the topic on which the controller will publish velocity commands, clk_topic should provide the simulation time (if using a simulation), and coll_topic should indicate whether a vehicle collision has occurred. Note that coll_topic only needs to be specified if collisions are being recorded for evaluation. The first three topics must be correctly specified in order for the vehicle to operate successfully.

## Classifier

The third section allows you to specify the folder of the classifier model, which should be the same as the output_dir specified in step 3d (and model_dir used in various other steps) of the main [README](https://github.com/sarapohland/parce/blob/main/README.md).

## Competency

The fourth section is where you should specify the parameters for the competency estimation model(s). You can choose whether to use both overall competency estimation and regional competency estimation. (Note that you can use one, both, or neither of these methods). If you choose to use overall competency estimation, you must specify the error threshold and location of the competency estimation model, which should be the decoder_dir specified in step 4f of the main [README](https://github.com/sarapohland/parce/blob/main/README.md). If you choose to use regional competency estimation, you again must specify the error threshold and location of the model, which should be the decoder_dir specified in step 5f. For the regional competency estimator, you should also specify the smoothing method (max or average pooling), along with the relevant parameters (i.e., kernel size, stride length, and padding) used to smooth the regional competency image. The mask parameters (height, base1, base2, and pad) indicate the region in the input image that is relevant to the vehicle from its current position. (This mask is a hexagon created by the create_mask function in [path_planning](https://github.com/sarapohland/parce/blob/main/navigation/control/planning/path_planning.py).) Finally, the turn_only parameter indicates whether a turning-based approach or trajectory-based approach is used for regional competency-aware navigation. 

## Controller

The fifth section details the parameters of the competency-aware contoller. You must specify the location of the dynamics model generated in step 6c of the main [README](https://github.com/sarapohland/parce/blob/main/README.md).

The parameters u_low and u_high indicate the control bounds, where lin is the linear velocity command (throttle) and ang is the turn rate command (steering). The cmd_rate is the rate at which the controller runs (in Hz) and cmd_time is the time (in seconds) over which a control sequence is executed for path-tracking. The backup_time and turn_time indicate the times (in seconds) used for the safety response when model competency is low. The error_time is currently not used.

For planning, you must specify the number of trajectories to be sampled (N), the time horizon of sampled trajectories (H), the maximum distance ahead the robot can travel during planning (max_goal), and the camera scaling factor (camera_scaling). You must also provide all of the reward weighting parameters, which indicate the relative importance of making progress towards the goal, minimizing the path length, and facing towards the goal. 

The controller uses LQR control to track the path chosen by the path planner. You must specify the diagonal entries of the state deviation cost matrix (Q) and the input deviation cost matrix (R). These parameters determine the relative weight of deviating from the reference states and inputs.

## Camera

Finally, the sixth section records the size of input images (number of channels, height in pixels, and width in pixels).
