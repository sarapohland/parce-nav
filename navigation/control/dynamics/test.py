import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from navigation.control.dynamics.utils import list_files, extract_bags

def plot_error(t, true, pred, filename):

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plot predicted and realized trajectory
    plt.subplot(1, 2, 1)
    xs, ys = true[:, 0], true[:, 1]
    plt.scatter(xs, ys, c='g', marker='.', s=2, label='True')
    xs, ys = pred[:, 0], pred[:, 1]
    plt.scatter(xs, ys, c='b', marker='.', s=2, label='Predicted')
    plt.title('True and Predicted Trajectories of Vehicle')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()

    # Plot model error for each state
    abs_error = np.abs(true - pred)
    plt.subplot(1, 2, 2)
    plt.plot(t, abs_error[:, 0], label='X Position (m)')
    plt.plot(t, abs_error[:, 1], label='Y Position (m)') 
    # plt.plot(t, abs_error[:, 2], label='Orientation (rad)') 
    # plt.plot(t, abs_error[:, 3], label='Linear Velocity (m/s)') 
    # plt.plot(t, abs_error[:, 4], label='Turn Rate (rad/s)') 
    plt.title('Vehicle Dynamics Model Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

    plt.savefig(filename)
    

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--model_file', type=str, default='dynamics/husky.p')
    parser.add_argument('--bag_files', type=str, default='bags/test/')
    parser.add_argument('--horizon', type=int, default='1')
    args = parser.parse_args()

    # Load vehicle dynamics model
    dynamics = pickle.load(open(args.model_file, 'rb'))

    # Load test data
    bag_files = list_files(args.bag_files)
    T, U, X = extract_bags(bag_files, dynamics.cmd_topic, dynamics.odom_topic, dynamics.vehicle_name)
    X = X.T[np.newaxis, :, :]
    U = U.T[np.newaxis, :, :]

    # Predict next states using dynamics model
    x_preds = []
    for i in range(0, len(T), args.horizon):
        x0 = X[:, i, :]
        us = U[:, i:i+args.horizon, :]
        xs = dynamics.predict_next_states(x0, us)
        x_preds.append(xs)
    X_PRED = np.concatenate(x_preds, axis=1)
    
    # Plot true state versus predicted and model error
    t, true, pred = T[1:], X[0, 1:, :], X_PRED[0, :-1, :]
    save_file = '{}-{}.png'.format(os.path.join(os.path.dirname(args.model_file), Path(args.model_file).stem), args.horizon)
    plot_error(t, true, pred, save_file)


if __name__ == '__main__':
    main()
