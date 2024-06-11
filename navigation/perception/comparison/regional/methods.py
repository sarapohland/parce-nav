import sys
import abc
import torch
import pickle
import torch.nn.functional as F
from tqdm import tqdm

import anomalib.models as models
import anomalib.data.utils as utils

class Detector:

    @abc.abstractmethod
    def map_scores(self, test_loader):
        pass

class DRAEM(Detector):
    
    def __init__(self, dataloader, epochs):
        self.name = 'DRAEM'
        try:
            self.estimator = pickle.load(open("tmp/draem.p", "rb"))
        except:
            self.estimator = models.draem.torch_model.DraemModel(sspcab=False)

            # Set optimization parameters
            augmenter = utils.Augmenter(None, beta=(0.1, 1.0))
            criterion = models.draem.loss.DraemLoss()
            optimizer = torch.optim.Adam(params=self.estimator.parameters(), lr=0.0001)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 600], gamma=0.1)

            # Train DRAEM model with simulated anomaly images
            for t in tqdm(range(epochs)):
                for X, y in dataloader:
                    original = F.interpolate(X, size=(224, 224))
                    augmented_image, anomaly_mask = augmenter.augment_batch(original)
                    reconstruction, prediction = self.estimator(augmented_image)
                    loss = criterion(original, reconstruction, anomaly_mask, prediction)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            pickle.dump(self.estimator, open("tmp/draem.p", "wb"))

    def map_scores(self, inputs, outputs):
        batch, _, height, width = inputs.size()
        self.estimator.training = False
        inputs = F.interpolate(inputs, size=(224, 224))
        outputs = self.estimator(inputs)[:,None,:,:]
        return -F.interpolate(outputs, size=(height, width)).detach()[0,0,:,:]

class FastFlow(Detector):
    
    def __init__(self, dataloader, epochs):
        self.name = 'Fast Flow'
        try:
            self.estimator = pickle.load(open("tmp/fastflow.p", "rb"))
        except:
            self.estimator = models.fastflow.torch_model.FastflowModel(
                input_size=(224, 224),
                backbone="resnet18",
                pre_trained=True,
                flow_steps=8,
                conv3x3_only=False,
                hidden_ratio=1.0
            )

            # Set optimization parameters
            criterion = models.fastflow.loss.FastflowLoss()
            optimizer = torch.optim.Adam(
                params=self.estimator.parameters(),
                lr=0.001, weight_decay=0.00001,
            )

            # Learn transformation b/t features and distribution
            for t in tqdm(range(epochs)):
                for X, y in dataloader:
                    inputs = F.interpolate(X, size=(224, 224))
                    hidden_variables, jacobians = self.estimator(inputs)
                    loss = criterion(hidden_variables, jacobians)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            pickle.dump(self.estimator, open("tmp/fastflow.p", "wb"))

    def map_scores(self, inputs, outputs):
        batch, _, height, width = inputs.size()
        self.estimator.training = False
        inputs = F.interpolate(inputs, size=(224, 224))
        map = self.estimator(inputs)
        return -F.interpolate(map, size=(height, width)).detach()[0,0,:,:]

class PaDiM(Detector):
    
    def __init__(self, dataloader):
        self.name = 'PaDiM'
        try:
            self.estimator = pickle.load(open("tmp/padim.p", "rb"))
        except:
            self.estimator = models.padim.torch_model.PadimModel(
                input_size=(224, 224),
                backbone="resnet18",
                pre_trained=True,
                layers=["layer1", "layer2", "layer3"],
                n_features=None,
            )

            # Fit a Gaussian model to the training embeddings
            embeddings = []
            for X, y in dataloader:
                inputs = F.interpolate(X, size=(224, 224))
                features = self.estimator.feature_extractor(inputs)
                embeddings.append(self.estimator.generate_embedding(features))
            embeddings = torch.vstack(embeddings)
            self.estimator.gaussian.fit(embeddings)
            pickle.dump(self.estimator, open("tmp/padim.p", "wb"))

    def map_scores(self, inputs, outputs):
        batch, _, height, width = inputs.size()
        self.estimator.training = False
        inputs = F.interpolate(inputs, size=(224, 224))
        map = self.estimator(inputs)
        return -F.interpolate(map, size=(height, width)).detach()[0,0,:,:]

class PatchCore(Detector):
    
    def __init__(self, dataloader):
        self.name = 'PatchCore'
        try:
            self.estimator = pickle.load(open("tmp/patchcore.p", "rb"))
        except:
            self.estimator = models.patchcore.torch_model.PatchcoreModel(
                input_size=(224, 224),
                backbone="wide_resnet50_2",
                pre_trained=True,
                layers=("layer2", "layer3"),
                num_neighbors=9,
            )

            # Subsample embedding based on coreset sampling and store to memory
            embeddings = []
            for X, y in dataloader:
                inputs = F.interpolate(X, size=(224, 224))
                embeddings.append(self.estimator(inputs))
            embeddings = torch.vstack(embeddings)
            self.estimator.subsample_embedding(embeddings, 0.1)
            pickle.dump(self.estimator, open("tmp/patchcore.p", "wb"))

    def map_scores(self, inputs, outputs):
        batch, _, height, width = inputs.size()
        self.estimator.training = False
        inputs = F.interpolate(inputs, size=(224, 224))
        map = self.estimator(inputs)[0]
        return -F.interpolate(map, size=(height, width)).detach()[0,0,:,:]

class Reverse(Detector):
    
    def __init__(self, dataloader, epochs):
        self.name = 'Reverse Distillation'
        try:
            self.estimator = pickle.load(open("tmp/reverse.p", "rb"))
        except:
            self.estimator = models.reverse_distillation.torch_model.ReverseDistillationModel(
                backbone="wide_resnet50_2",
                pre_trained=True,
                layers=("layer1", "layer2", "layer3"),
                input_size=(224,224),
                anomaly_map_mode=models.reverse_distillation.anomaly_map.AnomalyMapGenerationMode.ADD,
            )

            # Set optimization parameters
            criterion = models.reverse_distillation.loss.ReverseDistillationLoss()
            optimizer = torch.optim.Adam(
                params=list(self.estimator.decoder.parameters()) + 
                        list(self.estimator.bottleneck.parameters()),
                lr=0.005, betas=(0.5, 0.99),
            )

            # Train bottleneck layer and decoder network
            for t in tqdm(range(epochs)):
                for X, y in dataloader:
                    inputs = F.interpolate(X, size=(224, 224))
                    loss = criterion(*self.estimator(inputs))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            pickle.dump(self.estimator, open("tmp/reverse.p", "wb"))

    def map_scores(self, inputs, outputs):
        batch, _, height, width = inputs.size()
        self.estimator.training = False
        inputs = F.interpolate(inputs, size=(224, 224))
        map = self.estimator(inputs)
        return -F.interpolate(map, size=(height, width)).detach()[0,0,:,:]
 
class RKDE(Detector):
    
    def __init__(self, dataloader):
        self.name = 'Region-Based KDE'
        try:
            self.estimator = pickle.load(open("tmp/rkde.p", "rb"))
        except:
            self.estimator = models.rkde.torch_model.RkdeModel(
                roi_stage=models.rkde.region_extractor.RoiStage.RCNN,
                roi_score_threshold=0.001,
                min_box_size=25,
                iou_threshold=0.3,
                max_detections_per_image=100,
                n_pca_components=16,
                feature_scaling_method=models.components.classification.FeatureScalingMethod.SCALE,
                max_training_points=40000,
            )
            
            # Fit a KDE model to the training embeddings
            embeddings = []
            for X, y in dataloader:
                rois = self.estimator.region_extractor(X)
                if rois.shape[0] == 0:
                    features = torch.empty((0, 4096))
                else:
                    features = self.estimator.feature_extractor(X, rois.clone())
                embeddings.append(features)
            embeddings = torch.vstack(embeddings)
            self.estimator.classifier.fit(embeddings)
            pickle.dump(self.estimator, open("tmp/rkde.p", "wb"))

    def map_scores(self, inputs, outputs):
        # Get scores and regions of interest (ROIs)
        self.estimator.training = False
        rois, scores = self.estimator(inputs)

        # Create anomaly map from ROIs
        map = torch.zeros_like(inputs)[:,0,:,:]
        for roi, score in zip(rois, scores):
            idx, x1, y1, x2, y2 = roi.int()
            map[idx,x1:x2,y1:y2] = score
        return -map[0,0,:,:]

class STFPM(Detector):
    
    def __init__(self, dataloader, epochs):
        self.name = 'Student-Teacher Feature Pyramid Matching'
        try:
            self.estimator = pickle.load(open("tmp/stfpm.p", "rb"))
        except:
            self.estimator = models.stfpm.torch_model.STFPMModel(
                input_size=(224,224),
                backbone="resnet18",
                layers=("layer1", "layer2", "layer3"),
            )

            # Set optimization parameters
            criterion = models.stfpm.loss.STFPMLoss()
            optimizer = torch.optim.SGD(
                params=self.estimator.student_model.parameters(),
                lr=0.4, momentum=0.9, dampening=0.0, weight_decay=0.001,
            )

            # Train the student model
            for t in tqdm(range(epochs)):
                for X, y in dataloader:
                    inputs = F.interpolate(X, size=(224, 224))
                    teacher_features, student_features = self.estimator.forward(inputs)
                    loss = criterion(teacher_features, student_features)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            pickle.dump(self.estimator, open("tmp/stfpm.p", "wb"))

    def map_scores(self, inputs, outputs):
        batch, _, height, width = inputs.size()
        self.estimator.training = False
        inputs = F.interpolate(inputs, size=(224, 224))
        map = self.estimator(inputs)
        return -F.interpolate(map, size=(height, width)).detach()[0,0,:,:]
 