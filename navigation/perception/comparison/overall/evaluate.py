import os
import json
import time
import torch
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd

from navigation.perception.networks.model import NeuralNet
from navigation.perception.datasets.setup_dataloader import setup_loader
from navigation.perception.comparison.overall.methods import *

torch.manual_seed(0)

   
def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('method', type=str)
    parser.add_argument('--test_data', type=str, default='lunar-nav')
    parser.add_argument('--model_dir', type=str, default='model/classify/')
    parser.add_argument('--decoder_dir', type=str, default='model/reconstruct/')
    parser.add_argument('--save_file', type=str, default='results/overall/parce.csv')
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
    
    elif args.method == 'softmax':
        estimator = Softmax()

    elif args.method == 'dropout':
        estimator = Dropout(model)

    elif args.method == 'ensemble':
        estimator = Ensemble(args.model_dir)

    elif args.method == 'temperature':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = Temperature(model, train_loader)

    elif args.method == 'kl':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = KLMatching(model, train_loader)

    elif args.method == 'entropy':
        estimator = Entropy(model)

    elif args.method == 'openmax':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = OpenMax(model, train_loader)

    elif args.method == 'energy':
        estimator = Energy(model)

    elif args.method == 'odin':
        estimator = ODIN(model)

    elif args.method == 'mahalanobis':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = Mahalanobis(model, train_loader)

    elif args.method == 'vim':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = ViM(model, train_loader)

    elif args.method == 'knn':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = kNN(model, train_loader)

    elif args.method == 'she':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = SHE(model, train_loader)

    elif args.method == 'dice':
        train_loader = setup_loader(args.test_data, val=True)
        estimator = DICE(model, train_loader)

    else:
        raise NotImplementedError('Unknown Method for Competency Estimation')

    # Create data loaders
    id_test_loader = setup_loader(args.test_data, test=True, batch_size=1)
    ood_test_loader = setup_loader(args.test_data, ood=True, batch_size=1)

    # Set up dictionary to store results
    results = {'label': [], 'pred': [], 'ood': [], 'score': [], 'time': []}

    # Collect data from ID test set
    for X, y in id_test_loader:
        # Get prediction from perception model
        output = model(X)

        # Estimate competency scores
        start = time.time()
        score = estimator.comp_scores(X, output.detach().numpy())
        lapsed = time.time() - start

        # Save results
        results['label'].append(y.detach().item())
        results['pred'].append(torch.argmax(output, dim=1).detach().item())
        results['ood'].append(0)
        results['score'].append(score[0])
        results['time'].append(lapsed)

    # Collect data from OOD test set
    for X, y in ood_test_loader:
        # Get prediction from perception model
        output = model(X)

        # Estimate competency scores
        start = time.time()
        score = estimator.comp_scores(X, output.detach().numpy())
        lapsed = time.time() - start
    
        # Save results
        results['label'].append(y.detach().item())
        results['pred'].append(torch.argmax(output, dim=1).detach().item())
        results['ood'].append(1)
        results['score'].append(score[0])
        results['time'].append(lapsed)

    # Save results to CSV file
    folder = os.path.dirname(args.save_file)
    if not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.DataFrame(results)
    df.to_csv(args.save_file, index=False)


if __name__ == "__main__":
    main()
