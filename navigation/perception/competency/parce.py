import os
import json
import sklearn
import numpy as np
import configparser
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from navigation.perception.networks.model import NeuralNet
from navigation.perception.datasets.setup_dataloader import setup_loader
from navigation.perception.utils.segment import segment, segment_pixels


# Probabilistic and Reconstruction-Based Competency Estimation
class PARCE:

    def __init__(self, method, train_data, model_dir, decoder_dir):

        self.method = method
        if not self.method == 'overall' and not self.method == 'regional':
            raise ValueError('Competency estimation method must be overall or regional.')
        
        # Load trained perception model
        with open(os.path.join(model_dir, 'layers.json')) as file:
            layer_args = json.load(file)
        self.model = NeuralNet(layer_args)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
        self.model.eval()

        # Load trained decoder model
        with open(os.path.join(decoder_dir, 'layers.json')) as file:
            layer_args = json.load(file)
        self.decoder = NeuralNet(layer_args)
        self.decoder.load_state_dict(torch.load(os.path.join(decoder_dir, 'model.pth')))
        self.decoder.eval()

        # Set device (CPU or GPU)
        self.set_device()

        # Set up dataset
        config_file = os.path.join(decoder_dir, 'train.config')
        train_config = configparser.RawConfigParser()
        train_config.read(config_file)
        batch_size_test = train_config.getint('training', 'batch_size_test')
        test_loader = setup_loader(train_data, batch_size=batch_size_test, val=True)

        # Get zscore threshold for competency estimation
        self.zscore = train_config.getfloat('competency', 'zscore')

        # Get segmentation parameters
        if self.method == 'regional':
            self.sigma = train_config.getfloat('segmentation', 'sigma')
            self.scale = train_config.getint('segmentation', 'scale')
            self.min_size = train_config.getint('segmentation', 'min_size')
            self.min_reco = train_config.getint('segmentation', 'min_reco')

        # Get smoothing parameters
        try:
            self.kernel = train_config.getint('smoothing', 'kernel')
            self.stride = train_config.getint('smoothing', 'stride')
            self.padding = train_config.getint('smoothing', 'padding')
            self.smoothing = train_config.getint('smoothing', 'method')
        except:
            pass

        # Get reconstruction loss for all test images
        all_losses, all_labels = self._get_losses(test_loader)

        # Estimate reconstruction loss distributions
        self.C = len(np.unique(all_labels))
        class_losses = self._get_distributions(all_losses, all_labels)

        # Save kernel density estimation (KDE) plots
        fig, ax = plt.subplots()
        for idx, losses in enumerate(class_losses):
            sns.kdeplot(data=losses, ax=ax, label='Class {}'.format(idx))
        ax.legend()
        plt.xlabel('Reconstruction Loss')
        plt.ylabel('Probability Density')
        plt.title('Reconstruction Loss Distributions')
        plt.savefig(os.path.join(decoder_dir, 'train_distributions.png'))

    def _mask_images(self, images, labels):
        all_segments, all_labels = [], []
        all_masked_imgs, all_orig_imgs = [], []
        for img, lbl in zip(images, labels):
            # Reformat input image
            np_img = img.numpy()
            np_img = np.squeeze(np_img * 255).astype(np.uint8)
            np_img = np.swapaxes(np.swapaxes(np_img, 0, 1), 1, 2)

            # Perform image segmentation
            segments = segment(np_img, self.sigma, self.scale, self.min_size)
            all_pixels = segment_pixels(segments)

            # Create a mask tensor for each segment
            for pixels in all_pixels:
                masked_img = img.clone()[None,:,:,:]
                masked_img[:,:, pixels[0, :], pixels[1, :]] = 1
                all_masked_imgs.append(masked_img)
                all_orig_imgs.append(img.clone()[None,:,:,:])
                all_labels.append(lbl)
                all_segments.append(pixels)

        return torch.vstack(all_masked_imgs), torch.vstack(all_orig_imgs), all_labels, all_segments

    # Get reconstruction loss for all images in dataloader
    def _get_losses(self, dataloader):
        all_losses, all_labels = [], []
        loss_func = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for X, y in dataloader:
                # Overall competency method
                if self.method == 'overall':
                    # Load input data and label
                    input = X.to(self.device)
                    all_labels.append(y.numpy().flatten())

                    # Get prediction from image decoder
                    try:
                        pred = self.decoder(input)
                    except:
                        z = self.model.get_feature_vector(input)
                        pred = self.decoder(z.to(self.device))

                    # Compute reconstruction loss
                    loss = torch.mean(loss_func(input, pred), dim=(1,2,3))
                    all_losses.append(loss.numpy())

                # Regional competency method
                elif self.method == 'regional':
                    # Load input data and label
                    input, true, labels, segments = self._mask_images(X, y)
                    input, true = input.to(self.device), true.to(self.device)
                    all_labels.append(np.vstack(labels).flatten())

                    # Get prediction from image decoder
                    try:
                        pred = self.decoder(input)
                    except:
                        z = self.model.get_feature_vector(input)
                        pred = self.decoder(z.to(self.device))

                    # Compute reconstruction loss
                    for seg_true, seg_input, seg_pred, segment in zip(true, input, pred, segments):
                        if np.shape(segment)[1] < self.min_reco: 
                            avg_loss = torch.tensor(float('nan'))
                        else:
                            loss_img = loss_func(seg_true, seg_pred)
                            avg_loss = torch.mean(loss_img[:, segment[0, :], segment[1, :]])
                        all_losses.append(avg_loss.numpy())

                else:
                    raise NotImplementedError('Unknown competency estimation method.')

        return np.hstack(all_losses), np.hstack(all_labels)
    
    # Estimate reconstruction loss distributions for each label
    def _get_distributions(self, all_losses, all_labels):
        class_losses = []
        self.class_means, self.class_stds = [], []
        for c in range(self.C):
            class_losses.append(all_losses[all_labels == c])
            self.class_means.append(np.nanmean(class_losses[-1]))
            self.class_stds.append(np.nanstd(class_losses[-1]))
        return class_losses
    
    # Set device (CPU or GPU)
    def set_device(self, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)
        self.decoder.to(self.device)

    # Compute reconstruction losses
    def comp_losses(self, inputs, outputs=None):
        loss_func = nn.MSELoss(reduction='none')

        # Overall competency method
        if self.method == 'overall':
            # Load input data
            input = inputs.to(self.device)

            # Get prediction from image decoder
            try:
                pred = self.decoder(input)
            except:
                z = self.model.get_feature_vector(input)
                pred = self.decoder(z.to(self.device))

            # Compute reconstruction loss
            losses = torch.mean(loss_func(input, pred), dim=(1,2,3))
            losses = losses.cpu().detach().numpy()

            if outputs is not None:
                return losses, outputs, None
            return losses, None

        # Regional competency method
        elif self.method == 'regional':
            # Load input data
            input, true, outputs, segments = self._mask_images(inputs, outputs)
            input, true = input.to(self.device), true.to(self.device)

            # Get prediction from image decoder
            try:
                pred = self.decoder(input)
            except:
                z = self.model.get_feature_vector(input)
                pred = self.decoder(z.to(self.device))

            # Compute reconstruction loss
            losses = []
            for seg_true, seg_pred, segment in zip(true, pred, segments):
                if np.shape(segment)[1] < self.min_reco:
                    avg_loss = torch.tensor(float('nan'))
                else:
                    loss_img = loss_func(seg_true, seg_pred)
                    avg_loss = torch.mean(loss_img[:, segment[0, :], segment[1, :]])
                losses.append(avg_loss.cpu())
            losses = torch.vstack(losses).detach().numpy()
            outputs = np.vstack(outputs)

            return losses, outputs, segments

        else:
            raise NotImplementedError('Unknown competency estimation method.')
        
    # Estimate probabilistic perception model competency
    def comp_scores(self, inputs, outputs):
        # Compute reconstruction loss
        losses, class_probs, _ = self.comp_losses(inputs, outputs)

        # Estimate probability that image is ID
        loss_probs = np.zeros((len(losses), self.C))
        for c in range(self.C):
            zvals = (losses - (2 * self.class_means[c])) / self.class_stds[c] - self.zscore
            loss_probs[:, c] = 1 - stats.norm.cdf(zvals).flatten()
        
        # Estimate probability that model is competent
        scores = np.sum(loss_probs * class_probs, axis=1)
        if self.method == 'overall':
            scores *= np.max(class_probs, axis=1)
        # Another method...
        # idxs = np.argmax(class_probs, axis=1)
        # max_class_probs = class_probs[range(0,len(class_probs)), idxs]
        # max_loss_probs  = loss_probs[range(0,len(loss_probs)), idxs]
        # scores = max_class_probs * max_loss_probs
        
        return scores, losses.flatten()
    
    def set_smoothing(self, kernel=None, stride=None, padding=None, method=None):
        if kernel is not None:
            self.kernel = kernel
        if stride is not None:
            self.stride = stride
        if padding is not None:
            self.padding = padding
        if method is not None:
            self.smoothing = method
    
    # Map regional competency estimates to pixels in image
    def map_scores(self, input, output):
        # Compute reconstruction loss
        losses, class_probs, segments = self.comp_losses(input, output)

        # Estimate probability that model is competent
        loss_probs = np.zeros((len(losses), self.C))
        for c in range(self.C):
            zvals = (losses - (2 * self.class_means[c])) / self.class_stds[c] - self.zscore
            loss_probs[:, c] = 1 - stats.norm.cdf(zvals).flatten()
        scores = np.sum(loss_probs * class_probs, axis=1)
        # scores *= np.max(class_probs, axis=1)
        scores = np.nan_to_num(scores, nan=np.nanmean(scores))

        # Create tensor of competency estimates per pixel
        height = int(torch.max(torch.Tensor([torch.max(segment[0,:]) for segment in segments])).item() + 1)
        width  = int(torch.max(torch.Tensor([torch.max(segment[1,:]) for segment in segments])).item() + 1)
        score_img = np.zeros((height, width))
        for segment, score in zip(segments, scores):
            score_img[segment[0, :], segment[1, :]] = score

        # Smooth competency estimates
        score_tensor = torch.from_numpy(score_img)[None,:,:]
        if self.smoothing == 'min':
            pooling = torch.nn.MaxPool2d(self.kernel, self.stride, self.padding)
            smooth_scores = -pooling(-score_tensor)[0,:,:]
        elif self.smoothing == 'avg':
            pooling = torch.nn.AvgPool2d(self.kernel, self.stride, self.padding, count_include_pad=False)
            smooth_scores = pooling(score_tensor)[0,:,:]
        else:
            smooth_scores = score_tensor[0,:,:]
        
        return smooth_scores
    