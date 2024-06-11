
import os
import abc
import json
import copy
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_ood.detector as oodd

from navigation.perception.networks.model import NeuralNet

class Detector:

    @abc.abstractmethod
    def comp_scores(self, inputs, outputs):
        pass

class Softmax(Detector):
    
    def __init__(self):
        self.name = 'Maximum Softmax'

    def comp_scores(self, inputs, outputs):
        # Get softmax probability of predicted class
        return np.max(outputs, axis=1)

class Dropout(Detector):

    def __init__(self, model, method='variance', prob=0.5, num_iterations=20):
        self.name = 'MC Dropout'
        
        # Apply dropout before last fully connected layer
        self.model = model
        self.model.layers.insert(-2, nn.Dropout(prob))
        self.model.net = nn.Sequential(*self.model.layers)

        # Define method for uncertainty estimation
        assert method in ['mean', 'variance', 'entropy']
        self.method = method
        self.num_iterations = num_iterations

    def comp_scores(self, inputs, outputs):
        # Determine predicted class
        preds = np.argmax(outputs, axis=1)

        # Save softmax probability fo each iteration
        probs = []
        for itr in self.num_iterations:
            out = self.model(inputs).detach()
            probs.append(out[:,preds])
        probs = np.stack(probs, -1)

        # Compute the average softmax probability
        if self.method == 'mean':
            return np.mean(probs, axis=-1).flatten()
       
        # Compute the variance of softmax probabilities
        elif self.method == 'variance':
            return -np.var(probs, axis=-1).flatten()
        
        # Compute the entropy of softmax probabilities
        elif self.method == 'entropy':
            return np.sum(probs * np.log(probs + 1e-16), axis=-1).flatten()

class Ensemble(Detector):

    def __init__(self, model_dir, method='variance'):
        self.name = 'Ensemble'

        # Load trained models
        with open(os.path.join(model_dir, 'layers.json')) as file:
            layer_args = json.load(file)

        self.models = []
        for file in os.listdir(model_dir):
            if not file.endswith('.pth'):
                continue
            model = NeuralNet(layer_args)
            model.load_state_dict(torch.load(os.path.join(model_dir, file)))
            model.eval()
            self.models.append(model)

        # Define method for uncertainty estimation
        assert method in ['mean', 'variance', 'entropy']
        self.method = method

    def comp_scores(self, inputs, outputs):
        # Determine predicted class
        preds = np.argmax(outputs, axis=1)

        # Save softmax probability of predicted class for each model
        probs = []
        for model in self.models:
            out = model(inputs).detach()
            probs.append(out[:,preds])
        probs = np.stack(probs, -1)

        # Compute the average softmax probability
        if self.method == 'mean':
            return np.mean(probs, axis=-1).flatten()
       
        # Compute the variance of softmax probabilities
        elif self.method == 'variance':
            return -np.var(probs, axis=-1).flatten()
        
        # Compute the entropy of softmax probabilities
        elif self.method == 'entropy':
            return np.sum(probs * np.log(probs + 1e-16), axis=-1).flatten()

class Temperature(Detector):

    def __init__(self, model, dataloader):
        self.name = 'Temperature Scaling'

        # Determine the optimal temperature value
        self.estimator = oodd.TemperatureScaling(model)
        outputs, labels = [], []
        for X, y in dataloader:
            out = model(X)
            outputs.append(out)
            labels.append(y)
        outputs = torch.vstack(outputs)
        labels = torch.vstack(labels).flatten()
        self.estimator.fit_features(outputs.long(), labels.long())
        # print('Optimal temperature: {}'.format(self.estimator.t.item()))

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict_features(torch.from_numpy(outputs))
        return -scores.detach().numpy()

class KLMatching(Detector):

    def __init__(self, model, dataloader):
        self.name = 'KL-Matching'

        # Estimate typical distributions for each class
        self.estimator = oodd.KLMatching(model)
        outputs, labels = [], []
        for X, y in dataloader:
            out = model(X)
            outputs.append(out)
            labels.append(y)
        outputs = torch.vstack(outputs)
        labels = torch.vstack(labels).flatten()
        self.estimator.fit_features(outputs, labels)

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict_features(torch.from_numpy(outputs))
        return scores.detach().numpy()

class Entropy(Detector):

    def __init__(self, model):
        self.name = 'Entropy'
        self.estimator = oodd.Entropy(model)
        
    def comp_scores(self, inputs, outputs):
        scores = self.estimator.score(torch.from_numpy(outputs))
        return -scores.detach().numpy()

class OpenMax(Detector):

    def __init__(self, model, dataloader):
        self.name = 'OpenMax'

        # Estimate typical distributions for each class
        self.estimator = oodd.OpenMax(model)
        outputs, labels = [], []
        for X, y in dataloader:
            out = model(X)
            outputs.append(out)
            labels.append(y)
        outputs = torch.vstack(outputs).detach()
        labels = torch.vstack(labels).flatten().detach()
        self.estimator.fit_features(outputs, labels)

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict_features(torch.from_numpy(outputs))
        return -scores.detach().numpy()

class Energy(Detector):

    def __init__(self, model):
        self.name = 'Energy Based'
        self.estimator = oodd.EnergyBased(model)
        
    def comp_scores(self, inputs, outputs):
        scores = self.estimator.score(torch.from_numpy(outputs))
        return -scores.detach().numpy()

class ODIN(Detector):

    def __init__(self, model):
        self.name = 'ODIN'
        self.estimator = oodd.ODIN(model)
        
    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return -scores.detach().numpy()

class Mahalanobis(Detector):

    def __init__(self, model, dataloader):
        self.name = 'Mahalanobis Distance'

        # Get feature extractor from model
        feature_extractor = copy.deepcopy(model)
        del feature_extractor.layers[-2:]
        feature_extractor.net = nn.Sequential(*feature_extractor.layers)

        # Estimate typical distributions for each class
        self.estimator = oodd.Mahalanobis(feature_extractor)
        features, labels = [], []
        for X, y in dataloader:
            z = model.get_feature_vector(X)
            features.append(z)
            labels.append(y)
        features = torch.vstack(features).detach()
        labels = torch.vstack(labels).flatten().detach()
        self.estimator.fit_features(features, labels)

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return -scores.detach().numpy()

class ViM(Detector):

    def __init__(self, model, dataloader):
        self.name = 'Virtual Logit Matching'

        # Get feature extractor from model
        feature_extractor = copy.deepcopy(model)
        del feature_extractor.layers[-2:]
        feature_extractor.net = nn.Sequential(*feature_extractor.layers)

        # Get parameters of last linear layer
        d = model.layers[-2].in_features
        W = model.layers[-2].weight
        b = model.layers[-2].bias

        # Compute principle subspace
        self.estimator = oodd.ViM(feature_extractor, d, W, b)
        features, labels = [], []
        for X, y in dataloader:
            z = model.get_feature_vector(X)
            features.append(z)
            labels.append(y)
        features = torch.vstack(features).detach()
        labels = torch.vstack(labels).flatten().detach()
        self.estimator.fit_features(features, labels)
        print('alpha: ', self.estimator.alpha)

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return scores.detach().numpy()

class kNN(Detector):

    def __init__(self, model, dataloader):
        self.name = 'k Nearest Neighbor'

        # Get feature extractor from model
        feature_extractor = copy.deepcopy(model)
        del feature_extractor.layers[-2:]
        feature_extractor.net = nn.Sequential(*feature_extractor.layers)

        # Fit nearest neighbor model
        self.estimator = oodd.KNN(feature_extractor)
        features, labels = [], []
        for X, y in dataloader:
            z = model.get_feature_vector(X)
            features.append(z)
            labels.append(y)
        features = torch.vstack(features).detach()
        labels = torch.vstack(labels).flatten().detach()
        self.estimator.fit_features(features, labels)

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return -scores[0].detach().numpy()

class SHE(Detector):

    def __init__(self, model, dataloader):
        self.name = 'Simplified Hopfield Energy'

        # Get feature extractor and classifier from model
        feature_extractor = copy.deepcopy(model)
        del feature_extractor.layers[-2:]
        feature_extractor.net = nn.Sequential(*feature_extractor.layers)
        
        classifier = copy.deepcopy(model)
        del classifier.layers[:-2]
        classifier.net = nn.Sequential(*classifier.layers)

        # Estimate typical distributions for each class
        self.estimator = oodd.SHE(feature_extractor, classifier)
        features, labels = [], []
        for X, y in dataloader:
            z = model.get_feature_vector(X)
            features.append(z)
            labels.append(y)
        features = torch.vstack(features).detach()
        labels = torch.vstack(labels).flatten().detach()
        self.estimator.fit_features(features, labels)

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return scores.detach().numpy()

class DICE(Detector):

    def __init__(self, model, dataloader):
        self.name = 'DICE'

        # Get feature extractor from model
        feature_extractor = copy.deepcopy(model)
        del feature_extractor.layers[-2:]
        feature_extractor.net = nn.Sequential(*feature_extractor.layers)

        # Get parameters of last linear layer
        W = model.layers[-2].weight
        b = model.layers[-2].bias

        # Compute principle subspace
        self.estimator = oodd.DICE(feature_extractor, W, b, p=0.7)
        features, labels = [], []
        for X, y in dataloader:
            z = model.get_feature_vector(X)
            features.append(z)
            labels.append(y)
        features = torch.vstack(features).detach()
        labels = torch.vstack(labels).flatten().detach()
        self.estimator.fit_features(features, labels)

    def comp_scores(self, inputs, outputs):
        scores = self.estimator.predict(inputs)
        return -scores.detach().numpy()