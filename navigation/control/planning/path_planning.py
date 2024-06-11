import os
import cv2
import json
import time
import torch
import pickle
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from navigation.perception.networks.model import NeuralNet

from navigation.control.utils.image_geometry import PinholeCameraModel
from navigation.control.utils.camera_info import SpotCameraInfo, HuskyCameraInfo, WartyCameraInfo
from navigation.control.utils.coordinate_frames import robot_to_world, world_to_robot
from navigation.control.planning.action_sampler import ActionSampler
from navigation.control.planning.rewards import *

class PathPlanner():

    def __init__(self, config_file):
        # Load XML config file
        tree = ET.parse(config_file)
        config = tree.getroot()

        # Initialize path planner
        self.read_params(config)
        self.get_vehicle_model(config)
        self.set_camera_model(config)
        self.set_action_sampler(config)
        self.load_models(config)

    def read_params(self, config):
        # Read environment parameters
        environment = config.find("environment")
        self.goal_thresh = float(environment.find("thresh").get("value"))
        
        # Read control parameters
        controller = config.find("controller")
        control = config.find("controller").find("control")
        self.u_low  = np.array([float(control.find("u_low").get("lin")),
                                float(control.find("u_low").get("ang"))])
        self.u_high = np.array([float(control.find("u_high").get("lin")),
                                float(control.find("u_high").get("ang"))])
        self.cmd_rate = float(control.find("cmd_rate").get("value"))
        self.cmd_time    = float(control.find("cmd_time").get("value"))
        self.backup_time = float(control.find("backup_time").get("value"))
        self.turn_time   = float(control.find("turn_time").get("value"))

        # Read planning parameters
        planner = controller.find("planning")
        self.N = int(planner.find("N").get("value"))
        self.H = int(planner.find("H").get("value"))
        self.x_max = float(planner.find("max_goal").get("xmax"))
        self.camera_scale = float(planner.find("camera_scaling").get("value"))
        self.y_max = (self.x_max + 1) * self.camera_scale

        # Read reward parameters
        rewards = controller.find("rewards")
        self.goal_weight  = float(rewards.find("goal").get("weight"))
        self.goal_alpha   = float(rewards.find("goal").get("alpha"))
        self.goal_alphaf  = float(rewards.find("goal").get("alphaf"))
        self.goalx_weight = float(rewards.find("goalx").get("weight"))
        self.goalx_alpha  = float(rewards.find("goalx").get("alpha"))
        self.goalx_alphaf = float(rewards.find("goalx").get("alphaf"))
        self.goaly_weight = float(rewards.find("goaly").get("weight"))
        self.goaly_alpha  = float(rewards.find("goaly").get("alpha"))
        self.goaly_alphaf = float(rewards.find("goaly").get("alphaf"))
        self.path_weight  = float(rewards.find("path").get("weight"))
        self.path_alpha   = float(rewards.find("path").get("alpha"))
        self.path_alphaf  = float(rewards.find("path").get("alphaf"))
        self.ang_weight   = float(rewards.find("angle").get("weight"))
        self.ang_alpha    = float(rewards.find("angle").get("alpha"))
        self.ang_alphaf   = float(rewards.find("angle").get("alphaf"))
        self.dist_weight  = float(rewards.find("dist").get("weight"))
        self.dist_alpha   = float(rewards.find("dist").get("alpha"))
        self.dist_alphaf  = float(rewards.find("dist").get("alphaf"))

    def get_vehicle_model(self, config):
        # Retrieve dynamics model of vehicle
        dynamics = config.find("controller").find("dynamics")  
        dynamics_file = dynamics.find("file").get("value")
        self.dynamics = pickle.load(open(dynamics_file, 'rb'))

    def set_camera_model(self, config):
        # Read vehicle parameters
        environment = config.find("environment")
        vehicle_name = environment.find("vehicle").get("name")
        self.vehicle_width = float(environment.find("vehicle").get("width"))
        self.vehicle_length = float(environment.find("vehicle").get("length"))

        # Set camera model
        if vehicle_name == 'husky':
            camera_info = HuskyCameraInfo()
        elif vehicle_name == 'spot':
            camera_info = SpotCameraInfo()
        elif vehicle_name == 'warty':
            camera_info = WartyCameraInfo()
        else:
            raise NotImplementedError('Warning: unknown vehicle model!')
        self.camera = PinholeCameraModel()
        self.camera.fromCameraInfo(camera_info)

    def set_action_sampler(self, config):
        self.action_sampler = ActionSampler(self.H, self.N, self.u_low, self.u_high)
        
    def load_models(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load trained classification model
        classifier = config.find("classifier")
        model_dir = classifier.find("model").get("value")
        with open(os.path.join(model_dir, 'layers.json')) as file:
            layer_args = json.load(file)
        self.model = NeuralNet(layer_args)
        self.model.load_state_dict(torch.load(model_dir + 'model.pth'))
        # self.model.to(self.device) # TODO: CHECK IF FASTER USING GPU
        self.model.eval()

        # Load overal competency estimator
        self.ovl_thresh = 0.0
        self.overall_comp = None
        self.use_ovl_comp = False
        overall = config.find("competency").find("overall")
        if overall.get("enabled") == 'True':
            self.use_ovl_comp = True
            reconstruct_dir = overall.find("decoder_dir").get("value")
            self.overall_comp = pickle.load(open(os.path.join(reconstruct_dir, 'parce.p'), 'rb'))
            self.ovl_thresh = float(overall.find("error_thresh").get("value"))
            self.overall_comp.set_device()

        # Load regional competency estimator
        self.mask = None
        self.reg_thresh = 0.0
        self.regional_comp = None
        self.use_reg_comp = False
        self.use_traj_resp = False
        regional = config.find("competency").find("regional")
        if regional.get("enabled") == 'True':
            self.use_reg_comp = True
            inpaint_dir = regional.find("decoder_dir").get("value")
            self.regional_comp = pickle.load(open(os.path.join(inpaint_dir, 'parce.p'), 'rb'))

            smooth_params = [int(regional.find("smooth").get("kernel")),
                                  int(regional.find("smooth").get("stride")),
                                  int(regional.find("smooth").get("pad")),
                                  regional.find("smooth").get("method")]
            self.regional_comp.set_smoothing(smooth_params[0], smooth_params[1], 
                                             smooth_params[2], smooth_params[3])
            
            turn_only = regional.find("turn_only").get("enabled")
            self.use_traj_resp = True if turn_only == 'False' else False
            self.reg_thresh = float(regional.find("error_thresh").get("value"))

            height = int(regional.find("mask").get("height"))
            base1 = int(regional.find("mask").get("base1"))
            base2 = int(regional.find("mask").get("base2"))
            pad = int(regional.find("mask").get("pad"))
            self.mask = self.create_mask(height, base1, base2, pad)
            self.regional_comp.set_device()

    def create_mask(self, height, base1, base2, pad):
        trapezoid_width = max(base1, base2)
        trapezoid = np.zeros((self.camera.height - pad, self.camera.width), dtype=int)

        for i in range(self.camera.height - pad - height, self.camera.height - pad):
            width = base1 + (base2 - base1) * (i - (self.camera.height - pad - height)) / (height - 1)
            start = (self.camera.width - width) // 2
            end = start + width
            trapezoid[i, int(start):int(end)] = 1

        rectangle = np.ones((pad, self.camera.width))
        mask = np.vstack((trapezoid, rectangle))

        # import matplotlib.pyplot as plt
        # plt.imshow(mask, cmap='coolwarm')
        # plt.show()

        return torch.from_numpy(mask)
    
    def choose_safe_actions(self, comp_map=None):
        # Determine which direction to turn in
        comp_map = comp_map.numpy()
        if comp_map is None or np.all(comp_map == 0):
            direction = random.choice(['left', 'right'])

        width = np.shape(comp_map)[1]
        left_regions = comp_map[:,:int(width/2)]
        right_regions = comp_map[:,int(width/2):]
        if np.sum(left_regions) > np.sum(right_regions):
            direction = 'left'
        else:
            direction = 'right'

        # Get safe action response
        return self.get_safe_response(direction)
    
    def get_safe_response(self, direction):
        # Backup for some time
        num_backup = int(self.backup_time * self.cmd_rate)
        backup_cmds = np.tile([-self.u_high[0] / 2, 0], (1, num_backup, 1))

        # Turn in desired direction
        num_turn = int(self.turn_time * self.cmd_rate)
        if direction == 'left':
            turn_cmds = np.tile([0, self.u_high[1]], (1, num_turn, 1))
        elif direction == 'right':
            turn_cmds = np.tile([0, self.u_low[1]], (1, num_turn, 1))
        else:
            raise NotImplementedError('Unknown direction for safety controller.')

        return np.concatenate((backup_cmds, turn_cmds), axis=1)
    
    def sample_paths(self, x0):
        # Sample sequence of actions
        all_cmds = self.action_sampler.sample_action_seq(self.cmd_rate)
        
        # Predict vehicle path from action sequences
        x0[:3] = 0.0
        x0 = np.tile(x0, (self.N, 1))
        paths = self.dynamics.predict_next_states(x0, all_cmds)
        paths = np.concatenate((x0[:,np.newaxis,:], paths), axis=1)

        # for path in paths:
        #     xs = [pos[0] for pos in path]
        #     ys = [pos[1] for pos in path]
        #     plt.scatter(xs, ys, marker='.', s=2)
        # plt.show()

        # Remove paths that exit valid region
        valid_indices = np.where(
            (paths[:, :, 0].min(axis=1) >= -0.1) &
            (paths[:, :, 0].max(axis=1) <= self.x_max) &
            (paths[:, :, 1].min(axis=1) >= -self.y_max) &
            (paths[:, :, 1].max(axis=1) <= self.y_max)
        )
        paths = paths[valid_indices]
        cmds = all_cmds[valid_indices]

        return paths, cmds

    def get_pos_pixels(self, pos):
        rdf = [-pos[1]+0.07, 0.25, pos[0]+1.0] # start at edge of image: 0.8
        uv = self.camera.project3dToPixel(rdf)
        return (int(uv[0]), int(uv[1]))

    def get_traj_pixels(self, pos_traj):
        img_traj = []
        for pos in pos_traj:
            uv = self.get_pos_pixels(pos)
            if np.isnan(uv[0]) or np.isnan(uv[1]):
                continue
            if uv[0] < 0 or uv[0] > self.camera.width-1:
                continue
            if uv[1] < 0 or uv[1] > self.camera.height-1:
                continue
            img_traj.append(uv)
        return np.array(img_traj)
    
    def evaluate_error_probs(self, paths, comp_map):
        if not self.use_traj_resp:
            return np.ones(len(paths))
        
        # Resize competency map
        pil_img = Image.fromarray(comp_map.numpy())
        pil_img = pil_img.resize((self.camera.width, self.camera.height))
        comp_map = np.array(pil_img)
        
        # Get trajectories of left and right side of vehicle
        x, y, theta = paths[:,:,0], paths[:,:,1], paths[:,:,2]
        l, w = self.vehicle_length, self.vehicle_width
        r_trajs = np.stack([x + l/2 * np.cos(theta) + w/2 * np.sin(theta), 
                             y + l/2 * np.sin(theta) - w/2 * np.cos(theta)], axis=2)
        l_trajs = np.stack([x + l/2 * np.cos(theta) - w/2 * np.sin(theta), 
                             y + l/2 * np.sin(theta) + w/2 * np.cos(theta)], axis=2)
        
        # Compute minimum competency for each trajectory
        probs = []
        for l_traj, r_traj in zip(l_trajs, r_trajs):
            l_traj = self.get_traj_pixels(l_traj)
            r_traj = self.get_traj_pixels(r_traj)
            all_probs = [np.min(comp_map[np.minimum(l[1], r[1]):np.maximum(l[1], r[1])+1, 
                                           np.minimum(l[0], r[0]):np.maximum(l[0], r[0])+1])
                                           for (l, r) in zip(l_traj, r_traj)]
            try:
                probs.append(np.min(all_probs))
            except:
                probs.append(np.nan)

        return np.array(probs)
    
    def compute_comp_map(self, image):
        # Get output of perception model
        _, c, h, w = np.shape(image)
        output = self.model(image).detach().numpy()
        pred = np.max(output, 1)

        # Compute competency estimates
        # a) full competency awareness
        if self.overall_comp and self.regional_comp:
            competency = self.overall_comp.comp_scores(image, output)
            regional = self.regional_comp.map_scores(image, output)
            comp_map = torch.ones((h, w)) if competency > self.ovl_thresh else regional.clone()
       
        # b) overall-only awareness
        elif self.overall_comp and not self.regional_comp:
            competency = self.overall_comp.comp_scores(image, output)
            regional = torch.ones((h, w))
            comp_map = torch.ones((h, w)) if competency > self.ovl_thresh else torch.zeros((h, w))
       
        # c) regional-only awareness
        elif not self.overall_comp and self.regional_comp:
            competency = 0.0
            regional = self.regional_comp.map_scores(image, output)
            comp_map = regional
        
        # d) no competency awareness
        else:
            competency = 1.0
            regional = torch.ones((h, w))
            comp_map = torch.ones((h, w))

        return competency, regional, comp_map
    
    def select_best_path(self, state, goal, valid_paths, valid_probs, valid_cmds=None):

        # Check whether any of the valid paths enter the goal radius
        robot_paths = robot_to_world(valid_paths, state[:3])
        dists_to_goal = np.linalg.norm(robot_paths[:,:,:2] - goal[:2], axis=2)
        goal_reaching = [np.any(dists < self.goal_thresh) for dists in dists_to_goal]
        paths_to_goal = robot_paths[goal_reaching]
        
        # If some paths enter goal, choose the fastest path to goal
        if len(paths_to_goal) > 0:
            valid_paths   = valid_paths[goal_reaching]
            dists_to_goal = dists_to_goal[goal_reaching]
            goal_timesteps = [next(idx for (idx, dist) in enumerate(dists) if (dist < self.goal_thresh)) for dists in dists_to_goal]
            fastest_idx = np.argmin(goal_timesteps)
            end_time = goal_timesteps[fastest_idx] + 5
            
            best_path = np.concatenate((paths_to_goal[fastest_idx, :end_time], valid_paths[fastest_idx, :end_time, 3:]), axis=1)
            best_prob = valid_probs[goal_reaching][fastest_idx]
            if valid_cmds is not None:
                best_cmds = valid_cmds[goal_reaching][fastest_idx, :end_time]

        # Otherwise, determine safe trajectory with highest reward
        else:
            rewards = self.goal_weight * normalize(goal_cost(robot_paths, goal, self.goal_alpha, self.goal_alphaf)) + \
                        self.goalx_weight * normalize(goalx_cost(robot_paths, goal, self.goalx_alpha, self.goalx_alphaf)) + \
                        self.goaly_weight * normalize(goaly_cost(robot_paths, goal, self.goaly_alpha, self.goaly_alphaf)) + \
                        self.path_weight * normalize(path_cost(robot_paths, goal, self.path_alpha, self.path_alphaf)) + \
                        self.ang_weight * normalize(heading_cost(robot_paths, goal, self.ang_alpha, self.ang_alphaf)) + \
                        self.dist_weight * normalize(distance_reward(robot_paths, goal, self.dist_alpha, self.dist_alphaf))
            best_idx = np.argmax(rewards)
            
            best_path = np.concatenate((robot_paths[best_idx], valid_paths[best_idx, :, 3:]), axis=1)
            best_prob = valid_probs[best_idx]
            if valid_cmds is not None:
                best_cmds = valid_cmds[best_idx]

        if valid_cmds is not None:
            return best_path, best_prob, best_cmds
        else:
            return best_path, best_prob

    def plan_path(self, image, state, goal, visualize=False):

        # Get map of model competency
        competency, regional, comp_map = self.compute_comp_map(image)

        # Check for safe regions in vicinity of robot
        masked_regions = torch.clone(comp_map)
        if self.mask is not None:
            masked_regions[self.mask == 0] = 1.0

        # Save results for visualization
        if visualize:
            self._goal = world_to_robot(goal, state[:3])
            self._competency = competency
            self._regional = regional
            self._comp_map = comp_map
            self._sampled_paths = None
            self._sampled_probs = None
            self._valid_paths = None
            self._valid_probs = None
            self._best_prob = None
            self._best_path = None

        # If there are no safe regions in vicinity of vehicle, immediately resort to safe response
        if not torch.any(masked_regions > self.reg_thresh):
            safe_cmds = self.choose_safe_actions(masked_regions)
            return None, safe_cmds[0]

        # If not usng trajectory response and there is unsafe region in vicinity of vehicle, resort to safe response
        elif not self.use_traj_resp and torch.any(masked_regions <= self.reg_thresh):
            safe_cmds = self.choose_safe_actions(masked_regions)
            return None, safe_cmds[0]

        # Otherwise, check for safe trajectories
        else:
            # Sample paths towards goal
            paths, cmds = self.sample_paths(np.copy(state))

            # Eliminate paths with high probability of error
            probs = self.evaluate_error_probs(paths, comp_map)
            valid_cmds  = np.copy(cmds)[probs > self.reg_thresh]
            valid_paths = np.copy(paths)[probs > self.reg_thresh]
            valid_probs = np.copy(probs)[probs > self.reg_thresh]

            # Save results for visualization
            if visualize:
                self._sampled_paths = paths
                self._sampled_probs = probs
                self._valid_paths = valid_paths
                self._valid_probs = valid_probs

            # If no safe trajectory is found, return safe response
            if len(valid_paths) == 0:
                x0 = np.array(state)[np.newaxis,:]
                safe_cmds = self.choose_safe_actions(masked_regions)
                return None, safe_cmds[0]
            
            # Otherwise, select the "best" safe trajectory
            best_path, best_prob, best_cmds = self.select_best_path(state, goal, valid_paths, valid_probs, valid_cmds)
            
            # Save results for visualization
            if visualize:
                self._best_prob = best_prob
                self._best_path = best_path

            num_cmds = int(self.cmd_time * self.cmd_rate)
            return best_path[:num_cmds+1], best_cmds[:num_cmds]