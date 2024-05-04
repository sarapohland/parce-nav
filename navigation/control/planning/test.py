import os
import json
import torch
import pickle
import string
import argparse
import torchvision
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from navigation.perception.networks.model import NeuralNet
from navigation.perception.datasets.setup_dataloader import setup_loader
from navigation.control.planning.path_planning import PathPlanner

torch.manual_seed(0)

def plot_trajs(img, planner, file=None):
    # Get data from planner
    score = planner._comp_map
    all_trajs = None if planner._sampled_paths is None else \
        [planner.get_traj_pixels(path) if path is not None else None for path in planner._sampled_paths]
    all_probs = planner._sampled_probs
    valid_trajs = None if planner._valid_paths is None else \
        [planner.get_traj_pixels(path) if path is not None else None for path in planner._valid_paths]
    valid_probs = planner._valid_probs
    best_traj = planner.get_traj_pixels(planner._best_path) if planner._best_path is not None else None
    best_prob = planner._best_prob
    
    # Plot estimated regional competency
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    plt.subplot(1, 4, 1)
    im = plt.imshow(score, cmap='coolwarm_r', vmin=0, vmax=1)
    plt.title('Estimated Competency')
    
    # Plot sampled trajectories
    plt.subplot(1, 4, 2)
    img = np.squeeze(img.numpy() * 255).astype(np.uint8)
    img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
    plt.imshow(img)
    if all_trajs is not None and all_probs is not None:
        norm = colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='coolwarm_r')
        for traj, prob in zip(all_trajs, all_probs):
            if len(traj) > 0:
                color = np.array(mapper.to_rgba(prob)).reshape(1,-1)
                xs = [pos[0] for pos in traj]
                ys = [pos[1] for pos in traj]
                plt.scatter(xs, ys, c=color, marker='.', s=2)
    plt.title('Sampled Trajectories')

    # Plot valid trajectories
    plt.subplot(1, 4, 3)
    plt.imshow(img)
    if valid_trajs is not None and valid_probs is not None:
        norm = colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='coolwarm_r')
        for traj, prob in zip(valid_trajs, valid_probs):
            if len(traj) > 0:
                color = np.array(mapper.to_rgba(prob)).reshape(1,-1)
                xs = [pos[0] for pos in traj]
                ys = [pos[1] for pos in traj]
                plt.scatter(xs, ys, c=color, marker='.', s=2)
    plt.title('Safe Trajectories')

    # Plot selected trajectory
    plt.subplot(1, 4, 4)
    plt.imshow(img)
    goal_pos = planner.get_pos_pixels(planner._goal)
    plt.plot(goal_pos[0], goal_pos[1], color='green', marker='*', markersize=12)
    if best_traj is not None and best_prob is not None:
        color = np.array(mapper.to_rgba(best_prob)).reshape(1,-1)
        xs = [pos[0] for pos in best_traj]
        ys = [pos[1] for pos in best_traj]
        plt.scatter(xs, ys, c=color, marker='.', s=10)
    plt.title('Selected Trajectory')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Estimated Competency', rotation=270, labelpad=20)
    
    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('config_file', type=str)
    parser.add_argument('--test_data', type=str, default='lunar-nav')
    parser.add_argument('--init_x', type=float, default=0.0)
    parser.add_argument('--init_y', type=float, default=0.0)
    parser.add_argument('--goal_x', type=float, default=20.0)
    parser.add_argument('--goal_y', type=float, default=0.0)
    parser.add_argument('--example', type=int, default=None)
    args = parser.parse_args()

    # Initialize planner
    x0 = np.array([args.init_x, args.init_y, 0.0, 0.0, 0.0])
    xG = np.array([args.goal_x, args.goal_y, 0.0])
    planner = PathPlanner(args.config_file)
    
    # Create data loaders
    id_test_loader = setup_loader(args.test_data, test=True, batch_size=1)
    ood_test_loader = setup_loader(args.test_data, ood=True, batch_size=1)

    # Visualize single example
    if args.example is not None:
        # Create folder to save results
        output_dir = 'results/examples/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, (X, _) in enumerate(ood_test_loader):
            if idx == args.example:
                break

        # Run planner
        import time
        start = time.time()
        planner.plan_path(X, x0, xG, True)
        print('Time to plan path: ', time.time() - start)

        # Plot trajectories
        file = os.path.join(output_dir, '{}_({},{}).png'.format(args.example, round(xG[0]), round(xG[1])))
        plot_trajs(X, planner, file)

    # Visualize all test examples
    else:
        # Create folders to save results
        output_dir = 'results/planning/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        id_dir = os.path.join(output_dir, 'id')
        if not os.path.exists(id_dir):
            os.makedirs(id_dir)
        ood_dir = os.path.join(output_dir, 'ood')
        if not os.path.exists(ood_dir):
            os.makedirs(ood_dir)

        # Visualize examples from OOD test set
        for idx, (X, _) in enumerate(ood_test_loader):
            # Run planner
            planner.plan_path(X, x0, xG, True)

            # Plot trajectories
            file = os.path.join(ood_dir, '{}.png'.format(idx))
            plot_trajs(X, planner, file)

        # Visualize examples from ID test set
        for idx, (X, _) in enumerate(id_test_loader):
            # Run planner
            planner.plan_path(X, x0, xG, True)

            # Plot trajectories
            file = os.path.join(id_dir, '{}.png'.format(idx))
            plot_trajs(X, planner, file)

if __name__ == "__main__":
    main()