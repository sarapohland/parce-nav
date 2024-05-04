import os
import json
import pickle
import argparse
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from navigation.perception.networks.model import NeuralNet
from navigation.control.planning.path_planning import PathPlanner
from navigation.control.utils.coordinate_frames import world_to_robot


def get_figure(img, competency, regional, sampled_paths, sampled_probs, valid_paths, valid_probs, best_path, best_prob):

    # Plot estimated regional competency
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    im = axs[0,0].imshow(regional, cmap='coolwarm_r', vmin=0, vmax=1)
    try:
        competency = competency[0]
    except:
        pass
    axs[0,0].title.set_text('Estimated Competency: {}'.format(np.round(competency, 2)))
    
    # Plot sampled trajectories
    axs[0,1].imshow(img)
    if sampled_paths is not None and sampled_probs is not None:
        norm = colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='coolwarm_r')
        for traj, prob in zip(sampled_paths, sampled_probs):
            if len(traj) > 0:
                color = np.array(mapper.to_rgba(prob)).reshape(1,-1)
                xs = [pos[0] for pos in traj]
                ys = [pos[1] for pos in traj]
                axs[0,1].scatter(xs, ys, c=color, marker='.', s=2)
    axs[0,1].title.set_text('Sampled Trajectories')

    # Plot valid trajectories
    axs[1,0].imshow(img)
    if valid_paths is not None and valid_probs is not None:
        norm = colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='coolwarm_r')
        for traj, prob in zip(valid_paths, valid_probs):
            if len(traj) > 0:
                color = np.array(mapper.to_rgba(prob)).reshape(1,-1)
                xs = [pos[0] for pos in traj]
                ys = [pos[1] for pos in traj]
                axs[1,0].scatter(xs, ys, c=color, marker='.', s=2)
    axs[1,0].title.set_text('Safe Trajectories')

    # Plot selected trajectory
    axs[1,1].imshow(img)
    if best_path is not None and best_prob is not None:
        color = np.array(mapper.to_rgba(best_prob)).reshape(1,-1)
        xs = [pos[0] for pos in best_path]
        ys = [pos[1] for pos in best_path]
        axs[1,1].scatter(xs, ys, c=color, marker='.', s=10)
    axs[1,1].title.set_text('Selected Trajectory')

    return fig, axs

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('pickle_file', type=str)
    parser.add_argument('--save_dir', type=str, default='results/visualization/images/')
    args = parser.parse_args()

    # Create folder to store images
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Read stored data
    data = pickle.load(open(args.pickle_file, 'rb'))
    config = data['config']
    times  = data['time']
    states = data['state']
    images = data['image']
    comps  = data['comp']
    regnls = data['regnl']
    cmaps  = data['cmap']
    goals  = data['goal']
    trajs  = data['trajs']

    # Initialize planner
    planner = PathPlanner(config)

    # Run through each time step in recorded data
    init_state = init_time = None
    valid_paths = best_path = None
    sampled_probs = valid_probs = best_prob = None
    competency = regional = comp_map = None
    for time, state, image, goal, traj, comp, reg, cmap in \
        zip(times, states, images, goals, trajs, comps, regnls, cmaps):

        # if init_time is not None and time-init_time < 38.7:
        #     continue

        # Get output of planner if in planning step
        if image is not None:
            if init_time is None:
                init_time = time
            init_state = np.copy(state[:3])

            # competency, regional, comp_map = planner.compute_comp_map(image)
            competency, regional, comp_map = comp, reg, cmap

            valid_paths = best_path = None
            if traj is not None:
                sampled_probs = planner.evaluate_error_probs(traj, comp_map)
                valid_paths = np.copy(traj)[sampled_probs > planner.reg_thresh]
                valid_probs = np.copy(sampled_probs)[sampled_probs > planner.reg_thresh]

                if len(valid_paths) > 0:
                    best_path, best_prob = planner.select_best_path(init_state, goal, valid_paths, valid_probs)

            img = np.squeeze(image.numpy() * 255).astype(np.uint8)
            img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
            sampled_paths = None if traj is None else \
                [planner.get_traj_pixels(path) if path is not None else None for path in traj]
            valid_paths = None if valid_paths is None else \
                [planner.get_traj_pixels(path) if path is not None else None for path in valid_paths]
            best_path = planner.get_traj_pixels(world_to_robot(best_path, init_state)) if best_path is not None else None
            
        if init_state is not None:
            # Plot planning output
            fig, axs = get_figure(img, competency, regional, sampled_paths, sampled_probs, valid_paths, valid_probs, best_path, best_prob)
    
            # Get state of vehicle w.r.t. planning position
            state = world_to_robot(state[:3].reshape((1,-1)), init_state)[0]
            state = planner.get_pos_pixels(state)
            axs[1,1].plot(state[0], state[1], color='green', marker='*', markersize=6)
            plt.savefig(os.path.join(args.save_dir, '{}.png'.format(round(time-init_time, 2))))
            plt.close()
    
if __name__ == "__main__":
    main()