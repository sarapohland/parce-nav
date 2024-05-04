import os
import pickle
import argparse

from navigation.control.dynamics.vehicle import Vehicle
from navigation.control.dynamics.utils import list_files

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('config_file', type=str)
    parser.add_argument('--model_file', type=str, default='dynamics/husky.p')
    parser.add_argument('--bag_files', type=str, default='bags/train/')
    parser.add_argument('--horizon', type=int, default='1')
    parser.add_argument('--linear', action='store_true')
    args = parser.parse_args()

    # Fit dynamics model from training data
    bag_files = list_files(args.bag_files)
    dynamics = Vehicle(args.config_file, bag_files, args.horizon, args.linear)
    pickle.dump(dynamics, open(args.model_file, 'wb'))

if __name__ == "__main__":
    main()
