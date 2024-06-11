import os
import json
import time
import torch
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from navigation.perception.networks.model import NeuralNet
from navigation.perception.datasets.setup_dataloader import setup_loader
from navigation.perception.comparison.regional.methods import *

torch.manual_seed(0)

   
def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('method', type=str)
    parser.add_argument('--test_data', type=str, default='lunar-nav')
    parser.add_argument('--model_dir', type=str, default='model/classify/')
    parser.add_argument('--decoder_dir', type=str, default='model/inpaint/')
    parser.add_argument('--save_file', type=str, default='results/regional/parce.csv')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    # Set device for evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")

    # Load trained perception model
    with open(os.path.join(args.model_dir, 'layers.json')) as file:
        layer_args = json.load(file)
    model = NeuralNet(layer_args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))
    model.eval()
    model.to(device)

    # Load trained competency estimator
    if args.method == 'parce':
        file = os.path.join(args.decoder_dir, 'parce.p')
        estimator = pickle.load(open(file, 'rb'))
        estimator.set_device(device)
        config_file = os.path.join(args.decoder_dir, 'train.config')
        train_config = configparser.RawConfigParser()
        train_config.read(config_file)
        estimator.zscore = train_config.getfloat('competency', 'zscore')
        estimator.set_smoothing(method='none')

    elif args.method == 'draem':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = DRAEM(train_loader, 10)

    elif args.method == 'fastflow':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = FastFlow(train_loader, 10)

    elif args.method == 'padim':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = PaDiM(train_loader)

    elif args.method == 'patchcore':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = PatchCore(train_loader)

    elif args.method == 'reverse':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = Reverse(train_loader, 10)

    elif args.method == 'rkde':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = RKDE(train_loader)

    elif args.method == 'stfpm':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = STFPM(train_loader, 10)
    
    else:
        raise NotImplementedError('Unknown Method for Competency Estimation')

    # Create data loaders
    id_test_loader = setup_loader(args.test_data, test=True, batch_size=1)
    ood_test_loader = setup_loader(args.test_data, ood=True, batch_size=1)

    # Create folder to dump score images
    folder = os.path.dirname(args.save_file)
    if not os.path.exists(folder):
        os.makedirs(folder)
    sub_folder = os.path.join(folder, Path(args.save_file).stem)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    # Set up dictionary to store results
    results = {'label': [], 'pred': [], 'ood': [], 'score': [], 'time': []}

    # Collect data from ID test set
    for batch, (X, y) in enumerate(id_test_loader):
        # Get prediction from perception model
        output = model(X)

        # Estimate competency score image
        start = time.time()
        score_img = estimator.map_scores(X, output.detach().numpy())[None,:,:]
        lapsed = time.time() - start

        # Dump score image
        filename = os.path.join(sub_folder, '{}.npz'.format(len(results['score'])))
        score_img.numpy().dump(filename)

        # Save results
        results['label'].append(y.detach().item())
        results['pred'].append(torch.argmax(output, dim=1).detach().item())
        results['ood'].append(0)
        results['score'].append(filename)
        results['time'].append(lapsed)

    # Load OOD segmentation labels
    segmentation = pickle.load(open(os.path.join(args.decoder_dir, 'ood_labels.p'), 'rb'))
    seg_labels = segmentation['labels']

    # Collect data from OOD test set
    for batch, (X, y) in enumerate(ood_test_loader):
        # Ignore data without labeled regions
        batch_size = len(X)
        these_labels = seg_labels[batch * batch_size : (batch + 1) * batch_size]
        if len(these_labels) < 1:
            break
        elif len(these_labels) < batch_size:
            X = X[:len(these_labels),:,:,:]
            y = y[:len(these_labels)]

        # Get prediction from perception model
        output = model(X)

        # Estimate competency score image
        start = time.time()
        score_img = estimator.map_scores(X, output.detach().numpy())[None,:,:]
        lapsed = time.time() - start
    
        # Dump score image
        filename = os.path.join(sub_folder, '{}.npz'.format(len(results['score'])))
        score_img.numpy().dump(filename)

        # Save results
        results['label'].append(y.detach().item())
        results['pred'].append(torch.argmax(output, dim=1).detach().item())
        results['ood'].append(1)
        results['score'].append(filename)
        results['time'].append(lapsed)

    # Save results to CSV file
    df = pd.DataFrame(results)
    df.to_csv(args.save_file, index=False)


if __name__ == "__main__":
    main()
