# Probabilistic and Reconstruction-Based Competency Estimation (PaRCE)

This is the codebase for the paper titled [PaRCE: Probabilistic and Reconstruction-Based Competency Estimation for Safe Navigation Under Perception Uncertainty](https://arxiv.org/abs/2409.06111). This README describes how to reproduce the results achieved in this paper. 

## 0) Set Up Codebase

### 0a. Clone this repo

Clone this repository:
```
git clone https://github.com/sarapohland/parce.git
```

### 0b. Set up the navigation package

It's recommended that you create an environment with Python 3.7:
```
conda create -n parce python=3.7
```

Then, in the main folder (`parce`), run the following command:
```
pip install -e .
```

If you want to run the controller using ROS, you will need to also install ROS packages:
```
conda install conda-forge::ros-conda-base
```

```
conda install conda-forge::ros-rospy
```

## 1) Set Up Lunar Simulation

### 1a. Clone lunar simulation repo

To run experiments in the lunar simulation, you first need to clone the lunar-sim repository:

```
git clone https://github.com/sarapohland/lunar-sim.git
```

### 1b. Initialize lunar simulation

Follow the [README](https://github.com/sarapohland/lunar-sim/blob/master/README.md) in that repository to setup the lunar simulation.

## 2) Download Training Dataset

### 2a. Download the dataset files

To replicate the results presented in the paper, download the Lunar-Nav dataset file from the `data` folder available [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link). Create a folder called `data` in the  main directory (`parce`) and a subfolder called `Lunar-Nav`. Place the dataset file you downloaded into this subfolder. If you simply want to use the default lunar dataset, you can skip to step 3. If you want to create a new dataset from the lunar simulation environment, proceed through the rest of the substeps in this section.

### 2b. Collect training bags

To collect bagfiles from the lunar sim, activate the lunar-sim environment, then launch the simulation:
```
roslaunch lunar_gazebo lunar_test.launch world:=train_world
```

Once the simulation is running, you can teleop the vehicle by running the following command in another terminal:
```
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```

It is recommended that you create a folder called `bags` in the main directory (`parce`), where bagfiles will be collected. To collect bags of data while controlling the vehicle, run the following from within the bags folder of a third terminal:
```
rosbag record /mast_camera/image_raw
```

### 2c. Setup training dataset

Extract the images from the bagfile using the export_images launch file in the perception utils folder:
```
roslaunch navigation/perception/utils/export_imgs.launch bag:=<bag_file>
```

The parameter bag_file should be the name/location of the bagfile you collected in step 2b. By default, this command will save images to a folder called ~/.ros/. You can move the images to a more accessible folder using the following command:
```
mv ~/.ros/frame*.jpg <new_folder_name>
```

Once the images are in this folder, you can create a custom dataset by following the procedures described in the following substeps.

### 2d. Set up directory structure

By default, datasets are assumed to be saved in the following structure:
|-- data  
&emsp;|-- dataset1  
&emsp;&emsp;|-- dataset.npz  
&emsp;&emsp;|-- images  
&emsp;&emsp;&emsp;|-- ID  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- OOD  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- unsorted  
&emsp;|-- dataset2  
&emsp;&emsp;|-- dataset.npz  
&emsp;&emsp;|-- images   
&emsp;&emsp;&emsp;|-- ID  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- OOD  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- unsorted 

The unsorted folder should contain in-distribution training images that have not been labeled, while the ID folder contains all labeled in-distribution images organized by their class labels. If you already have a labeled dataset, you can organize them in the ID folder and skip to step 2f. If you only have unlabeled data, you can place it all in the unsorted folder and proceed to step 2e. The OOD folder should contain all out-of-distribution images. If this data is labeled, it can be orgnized into its class labels. If it is unlabeled, you can place it all into the same subfolder within the OOD folder. A dataset that has already been set up (following step 2f) will be saved in a compressed NumPy file called dataset.npz in the main dataset folder.

### 2e. Cluster unlabeled data

If you have labeled data, skip to the next step. If you have unlabeled in-distribution data saved in the unsorted directory, you can cluster these images using the create_dataset script:

```
python navigation/perception/datasets/create_dataset.py <path_to_dataset> --cluster_data
```

This command will cluster the unsorted images and save them in subfolders within the ID folder.

### 2f. Save custom dataset

Once you have existing classes of in-distribution data, you can save a dataset of training, test, and ood data using the create_dataset script:

```
python navigation/perception/datasets/create_dataset.py <path_to_dataset> --save_data
```

Note that this step can be combined with the previous one. By separating these two steps, you can validate the generated clusters before saving your dataset. You can also use to height and width flags to resize your images if desired. This script will save a compressed NumPy file called dataset.npz in your dataset directory.

### 2g. Update dataloader setup script

Use the existing cases in the setup_dataloader script to enable the use of your custom dataset. You will need to add a section to the get_class_names, get_num_classes, and the setup_loader functions.

## 3) Generate Classification Model

### 3a. Download the model files

Download the models folder from [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link) and place it in the main directory (`parce`). This folder contains the default lunar classification model for navigation, along with the model architectures and training parameters used to train the model. If you want to modify the configurations to train new models, go through the remaining steps in this section. To evaluate the classification model, see substep 3e. Otherwise, you can skip to step 4. 

### 3b. Define the model architecture

Create a JSON file defining your model architecture using the example given in `navigation/perception/networks/classification/layers.json`. Currently, you can define simple model architectures composed of convolutional, pooling, and fully-connected (linear) layers with linear, relu, hyperbolic tangent, sigmoid, and softmax activation functions. You can also perform 1D and 2D batch normalization and add a flattening layer in between other layers. For convolutional layers, you must specify the number of input and output channels and the kernel size. You can optionally provide the stride length and amount of zero padding. For pooling layers, you must specify the pooling function (max or average) and the kernel size. Finally, for fully-connected layers, you must specify the number of input and output nodes.

### 3c. Define the training parameters

Create a configuration file defining your training parameters using the example given in `navigation/perception/networks/classification/train.config`. You must specify the optimizer (sgd or adam), as well as the relevant optimizer parameters. Here you should also specify the desired loss function, number of epochs, and training/test batch sizes.

### 3d. Train the classification model

You can train your model using the train script in the networks classification folder:

```
python navigation/perception/networks/classification/train.py --train_data lunar-nav --output_dir models/lunar-nav/classify/ --train_config models/lunar-nav/classify/train.config --network_file models/lunar-nav/classify/layers.json
```

The argument train_data is used to indicate which dataset should be used to train your classification model, which should be lunar-nav if you are using the default training dataset. The argument output_dir is used to define where your trained classification model will be saved. (This is `models/lunar-nav/classify` for the default model downloaded in 3a.) The arguments network_file and train_config are used to specify the location of your model architecture JSON file (created in 3b) and training parameter config file (created in 3c). You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 3e. Evaluate the classification model

You can evaluate your model using the test script in the networks classification folder:

```
python navigation/perception/networks/classification/test.py --test_data lunar-nav --model_dir models/lunar-nav/classify/
```

The argument test_data is used to indicate which dataset should be used to evaluate your classification model, which should be lunar-nav if you are using the default evaluation dataset. The argument model_dir is used to specify where your trained classification model was saved. This should be the same location defined as the output_dir in step 3d. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will save a confusion matrix to the model_dir directory.

## 4) Design Overall Competency Estimator

### 4a. Download the model files

If you have not done so already, download the models folder from [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link) and place it in the main directory (`parce`). This folder contains the default lunar image reconstruction model, along with the model architectures and training parameters used to train the model. The trained overall competency estimator used in the paper is also contained in this folder. If you want to modify the configurations to train new models, go through the remaining steps in this section. To evaluate the reconstruction model, see substep 4e. To evaluate the overall competency estimator, see 4g. To visualize examples of overall model competency estimates, see substep 4h. Finally, to compare our method to existing methods for competency estimation, see substep 4i. Otherwise, you can skip to step 5. 

### 4b. Define the model architecture

Create a JSON file defining your model architecture using the example given in `navigation/perception/networks/reconstruction/layers.json`. The reconstruction model used by the overall competency estimator is meant to reconstruct the input image. Currently, you can define simple model architectures composed of convolutional, pooling, transposed convolutional, unsampling, and fully-connected (linear) layers with linear, relu, hyperbolic tangent, sigmoid, and softmax activation functions. You can also perform 1D and 2D batch normalization and add an unflattening layer in between other layers. For transposed convolutional layers, you must specify the number of input and output channels and the kernel size. You can optionally provide the stride length and the input/output zero padding. For unsampling layers, you must specify the scale factor or the target output size. If the unsampling mode is not specified, then the 'nearest' unsampling technique will be used. For fully-connected layers, you must specify the number of input and output nodes. Finally, for unflattening the number of output channels, as well as the resulting height and width.

### 4c. Define the training parameters

Create a configuration file defining your training parameters using the example given in `navigation/perception/networks/reconstruction/train.config`. You must specify the optimizer (sgd or adam), as well as the relevant optimizer parameters. Here you should also specify the desired loss function, number of epochs, and training/test batch sizes.

### 4d. Train the reconstruction model

To train the image reconstruction model, you can use the train script in the networks reconstruction folder:

```
python navigation/perception/networks/reconstruction/train.py reconstruct --architecture autoencoder --train_data lunar-nav --model_dir models/lunar-nav/classify/ --output_dir models/lunar-nav/reconstruct/ --train_config models/lunar-nav/reconstruct/train.config --network_file models/lunar-nav/reconstruct/layers.json
```

The argument train_data is used to indicate which dataset should be used to train your classification model, which should be lunar-nav if you are using the default training dataset. The argument model_dir is used to specify where your trained classification model was saved. This should be the same location defined as the output_dir in step 3d. The argument output_dir is used to define where your trained reconstruction model will be saved. (This is `models/lunar-nav/reconstruct` for the default model.) The arguments network_file and train_config are used to specify the location of your model architecture JSON file (created in 4b) and training parameter config file (created in 4c). You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 4e. Evaluate the reconstruction model

To evaluate the image reconstruction model, you can use the test script in the networks reconstruction folder:

```
python navigation/perception/networks/reconstruction/test.py reconstruct --architecture autoencoder --test_data lunar-nav --model_dir models/lunar-nav/classify/ --decoder_dir models/lunar-nav/reconstruct/
```

The argument test_data is used to indicate which dataset should be used to evaluate your reconstruction model, which should be lunar-nav if you are using the default evaluation dataset. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will save several figures (displaying the original and reconstructed images, along with the reconstruction loss) to a folder called `reconstruction` in decoder_dir.

### 4f. Train the overall competency estimator

You can train a competency estimator for your model using the train script in the competency folder:

```
python navigation/perception/competency/train.py overall --train_data lunar-nav --model_dir models/lunar-nav/classify/ --decoder_dir models/lunar-nav/reconstruct/
```

The argument train_data is used to indicate which dataset should be used to train the overall competency estimator, which should be lunar-nav if you are using the default training dataset. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 4g. Evaluate the overall competency estimator

You can evaluate your competency estimator using the test script in the competency folder:

```
python navigation/perception/competency/test.py overall --test_data lunar-nav --model_dir models/lunar-nav/classify/ --decoder_dir models/lunar-nav/reconstruct/
```

The argument test_data is used to indicate which dataset should be used to evaluate the overall competency estimator, which should be lunar-nav if you are using the default evaluation dataset. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will generate plots of the reconstruction loss distributions and probabilistic competency estimates for the correctly classified and misclassified in-distribution data, as well as the out-of-distribution data, and save them to the decoder_dir directory.

### 4h. Visualize the overall competency estimates

You can visualize the overall competency estimates for each test image using the visualize script in the competency folder:

```
python navigation/perception/competency/visualize.py overall --test_data lunar-nav --model_dir models/lunar-nav/classify/ --decoder_dir models/lunar-nav/reconstruct/
```

The argument test_data is used to indicate which dataset should be used for visualization, which should be lunar-nav if you are using the default evaluation dataset. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to visualize the model estimates using a GPU. This script will save figures of the input image and competency score to subfolders (correct, incorrect, and ood) in a folder called `competency` in decoder_dir.

### 4i. Competency overall competency scores to existing methods

We compare our overall competency scores to a number of uncertainty quantification and OOD detection methods, which are implemented in `navigation/perception/comparison/overall/methods.py`. We collect results for each method individually using the evaluate script in that folder:

```
python navigation/perception/comparison/overall/evaluate.py parce --save_file results/overall/parce.csv
```

In the command above, you can replace `parce` with each of the available methods (softmax, temperature, entropy, energy, dropout, ensemble, odin, kl, openmax, mahalanobis, knn, or dice) and select the CSV file where you would like to save the results for the partcicular method. You can optionally use the use_gpu flag to run evaluations on your GPU. To change the dataset, classification model, and reconstruction model from their defaults, use the test_data, model_dir, and decoder_dir arguments. Note that when evaluating the ensemble method, you will need to specify the model_dir that contains all of the model path files for the ensemble. (The ensemble we use is provided in the models folder [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link).) This command will save a CSV file of results to the file indicated by the save_file argument.

After running evaluations for each of the methods, you can compare them using the compare script in the same folder:

```
python navigation/perception/comparison/overall/compare.py results/overall/
```

You should replace `results/overall/` with the folder where all of the evaluation files are stored. This command will pull all of the CSV files from the given folder, read the results, calculate a number of performance metrics for each method, and print a table comparing the metrics to the terminal. It will also save figures of the score distributions for each method to the provided folder, along with ROC curves.

## 5) Design Regional Competency Estimator

### 5a. Download the model files

If you have not done so already, download the models folder from [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link) and place it in the main directory (`parce`). This folder contains the default lunar image inpainting model, along with the model architectures and training parameters used to train the model. The trained regional competency estimator used in the paper is also contained in this folder, along with labels for the segmented OOD dataset provided. If you want to modify the configurations to train new models, go through the remaining steps in this section. To evaluate the image inpainting model, see substep 5e. To evaluate the regional competency estimator, see 5h. To visualize examples of the regional competency maps, see substep 5i. Finally, to compare our method to existing methods for anomaly localization, see substep 5j. Otherwise, you can skip to step 6. 

### 5b. Define the model architecture

Create a JSON file defining your model architecture using the example given in `navigation/perception/networks/reconstruction/layers.json`. The image inpainting model used by the regional competency estimator is meant to reconstruct the input image from the feature vector used by the perception model to make a classification decision on a segmented and masked image. Currently, you can define simple model architectures composed of convolutional, pooling, transposed convolutional, unsampling, and fully-connected (linear) layers with linear, relu, hyperbolic tangent, sigmoid, and softmax activation functions. You can also perform 1D and 2D batch normalization and add an unflattening layer in between other layers. For transposed convolutional layers, you must specify the number of input and output channels and the kernel size. You can optionally provide the stride length and the input/output zero padding. For unsampling layers, you must specify the scale factor or the target output size. If the unsampling mode is not specified, then the 'nearest' unsampling technique will be used. For fully-connected layers, you must specify the number of input and output nodes. Finally, for unflattening the number of output channels, as well as the resulting height and width.

### 5c. Define the training parameters

Create a config file defining your training parameters using the example given in `navigation/perception/networks/reconstruction/train.config`. You must specify the optimizer (sgd or adam), as well as the relevant optimizer parameters. Here you should also specify the desired loss function, number of epochs, and training/test batch sizes. To train the inpainting model, you must also specify the segmentation parameters: sigma, scale, and min_size.

### 5d. Train the inpainting model

To train the image inpainting model, you can use the train script in the networks reconstruction folder:

```
python navigation/perception/networks/reconstruction/train.py inpaint --architecture autoencoder --train_data lunar-nav --model_dir models/lunar-nav/classify/ --output_dir models/lunar-nav/inpaint/ --train_config models/lunar-nav/inpaint/train.config --network_file models/lunar-nav/inpaint/encoder.json --init_model models/lunar-nav/reconstruct/
```

The argument train_data is used to indicate which dataset should be used to train your classification model, which should be lunar-nav if you are using the default training dataset. The argument model_dir is used to specify where your trained classification model was saved. This should be the same location defined as the output_dir in step 3d. The argument output_dir is used to define where your trained image inpainting model will be saved. (This is `models/lunar-nav/inpaint` for the default model.) The arguments network_file and train_config are used to specify the location of your model architecture JSON file (created in 5b) and training parameter config file (created in 5c). You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 5e. Evaluate the inpainting model

To evaluate the image inpainting model, you can use the test script in the networks reconstruction folder:

```
python navigation/perception/networks/reconstruction/test.py inpaint --architecture autoencoder --test_data lunar-nav --model_dir models/lunar-nav/classify/ --decoder_dir models/lunar-nav/inpaint/
```

The argument test_data is used to indicate which dataset should be used to evaluate your image inpainting model, which should be lunar-nav if you are using the default evaluation dataset. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained inpainting model was saved. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will save several figures (displaying the original, masked, and reconstructed images, along with the reconstruction loss) to a folder called reconstruction in decoder_dir.

### 5f. Train the regional competency estimator

You can train a competency estimator for your model using the train script in the competency folder:

```
python navigation/perception/competency/train.py regional --train_data lunar-nav --model_dir models/lunar-nav/classify/ --decoder_dir models/lunar-nav/inpaint/
```

The argument train_data is used to indicate which dataset should be used to train the regional competency estimator, which should be lunar-nav if you are using the default training dataset. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained image inpainting model was saved. You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 5g. Generate true labels of familiar/unfamiliar regions in image

To evaluate the performance of the regional competency estimator, you should generate labels of the regions in the OOD images that are familiar or unfamiliar to the perception model using the create_labels script:

```
python navigation/perception/utils/create_labels.py --test_data lunar-nav --decoder_dir models/lunar-nav/inpaint/
```

Each image in the OOD set of the test_data dataset will be segmented, and you will be shown each segmented region with the prompt: "Does this segment contain a structure not present in the training set?" Answering yes (y) will indicate that this region is unfamiliar to the model, while answering no (n) will indicate that it is familiar. These responses will be saved to a pickle file called ood_labels.p in the decoder_dir directory. Note that you can also review these labels using the test flag and begin relabeling from the middle of the OOD set using the start_idx parameter.

### 5h. Evaluate the regional competency estimator

You can evaluate your competency estimator using the test script in the competency folder:

```
python navigation/perception/competency/test.py regional --test_data lunar-nav --model_dir models/lunar-nav/classify/ --decoder_dir models/lunar-nav/inpaint/
```

The argument test_data is used to indicate which dataset should be used to evaluate the regional competency estimator, which should be lunar-nav if you are using the default evaluation dataset. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained image inpainting model was saved. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will generate plots of the reconstruction loss distributions and probabilistic competency estimates for both the familiar and unfamiliar regions in the out-of-distribution images (as determined in the previous step), as well as all of the regions in the in-distribution images, and save them to the decoder_dir directory.

### 5i. Visualize the regional competency estimates

You can visualize the regional competency maps for each test image using the visualize script in the competency folder:

```
python navigation/perception/competency/visualize.py regional --test_data lunar-nav --model_dir models/lunar-nav/classify/ --decoder_dir models/lunar-nav/inpaint/
```

The argument test_data is used to indicate which dataset should be used for visualization, which should be lunar-nav if you are using the default evaluation dataset. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained image inpainting model was saved. You can optionally use the use_gpu flag if you want to visualize the competency maps using a GPU. This script will save figures of the input image, true labeled image (with the labels from 5g), and the regional competency model predictions to subfolders (id and ood) in a folder called competency in decoder_dir.

### 5j. Competency regional competency maps to existing methods

We compare our regional competency maps to a number of anomaly detection and localization methods, which are implemented in `navigation/perception/comparison/regional/methods.py`. We collect results for each method individually using the evaluate script in that folder:

```
python navigation/perception/comparison/regional/evaluate.py parce --save_file results/regional/parce.csv
```

In the command above, you can replace `parce` with each of the available methods (draem, fastflow, padim, patchcore, reverse, rkde, or stfpm) and select the CSV file where you would like to save the results for the partcicular method. You can optionally use the use_gpu flag to run evaluations on your GPU. To change the dataset, classification model, and inpainting model from their defaults, use the test_data, model_dir, and decoder_dir arguments. This command will save a CSV file of results to the file indicated by the save_file argument, along with a folder containing saved competency maps.

After running evaluations for each of the methods, you can compare them using the compare script in the same folder:

```
python navigation/perception/comparison/regional/compare.py results/regional/
```

You should replace `results/regional/` with the folder where all of the evaluation files are stored. You should also specify the location of the saved OOD region labels using the decoder_dir argument (if they're not in the default location). This command will pull all of the CSV and data files from the given folder, read the results, calculate a number of performance metrics for each method, and print a table comparing the metrics to the terminal. It will also save figures of the score distributions for each method to the provided folder, along with ROC curves.

You can also visualize the competency maps for each evaluated method using the visualize script:

```
python navigation/perception/comparison/regional/visualize.py results/regional/ --save_dir results/regional/visualize/ --example 0
```

You should replace `results/regional/` with the folder where all of the evaluation files are stored. This command will pull all of the CSV and data files from the given folder, visualize the generated competency maps, and save the figures to the folder specified by save_dir.

## 6) Estimate Vehicle Dynamics Model

### 6a. Download the model file

To replicate the results presented in the paper, download the husky file from the `dynamics` folder available [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link). Create a folder called `dynamics` in the  main directory (`parce`) and place the dataset file you downloaded into this folder. If you simply want to use the default Husky dynamics model, you can skip to step 7. If you want to train a new dynamics model, proceed through the rest of the substeps in this section.

### 6b. Define the controller parameters

To estimate a model of the vehicle dynamics, you will need to collect data of the vehicle running under a random walk controller. Before doing so, you must specify the parameters for the random walk controller in a configuration file like the examples given in `navigation/control/configs/`.

For now, you only need to specify use_sim, vehicle, and time_limit under the environment section. The parameter use_sim indicates whether trials are being run in simulation or on a physical platform, vehicle specifies which vehicle model should be used (e.g., husky, spot, warty), and time_limit indicates the amount of time allocated for vehicle data collection. You also need to specify the vehicle odometry topic (odom_topic) and velocity command topic (cmd_topic) under the topics section. Finally, under the controller section, you must specify cmd_rate, u_low, and u_high. The parameters u_low and u_high are the minimum and maximum velocities (lin is linear velocity and ang is turn rate) of an action, and the cmd_rate is the rate at which the control runs (in Hz).

### 6c. Collect data under random walk controller

To collect data of the vehicle dynamics in the lunar simulation, first bring up the lunar simulation:

```
roslaunch lunar_gazebo lunar_test.launch world:=train_world
```

Then from another terminal, move into the folder where you would like to save bag files. Then, run the following command to start collecting data:
```
python navigation/control/dynamics/collect_data.py <xml_file>
```

The xml_file should be the same one chosen/created in the previous step. This will save bag files of the velocity commands and vehicle odometry to your working directory. Note that you should collect multiple bag files for training and evaluating your dyanmics model.

### 6d. Train vehicle dynamics model

To train a model of the vehicle dynamics from the data you collected in the previous step, you can use the train script in the dynamics folder:
```
python navigation/control/dynamics/train.py <xml_file> --model_file dynamics/husky.p --bag_files bags/train/
```

The xml_file should be the same one chosen/created in the step 6a. The argument model_file indicates where the model of the vehicle dynamics will be saved, and bag_files indicates the folder where the training bag files were saved in step 6b. You can optionally use the horizon argument to indicate how many steps ahead you would like to predict the state of the vehicle when determining the parameters of the non-linear dynamics model.

### 6e. Evaluate vehicle dynamics model

To evaluate your model of the vehicle dynamics, you can use the test script in the dynamics folder:
```
python navigation/control/dynamics/test.py --model_file dynamics/husky.p --bag_files bags/test/
```

The model_file should be the same file used for training in step 6c. The argument bag_files indicates the folder where the test bag files were saved in step 6b. This command will save plots of the true and predicted vehicle state over time, along with plots of the prediction error, to the same location where your dynamics model was saved. You can optionally use the horizon argument to indicate how many steps ahead you would like to predict the state of the vehicle.

## 7) Design Competency-Aware Path Planner

### 7a. Define the planning parameters

To design the competency-aware path planning algorithm, you must specify the parameters for the planner in a configuration file like the examples given in `navigation/control/configs/`. If you simply want to run with the default planning parameters used for evaluation, you can use the config files provided in this folder, but you will need to change the file locations to match your (absolute) file paths. More details on the configurations are provided in the configs [README](https://github.com/sarapohland/parce/tree/main/navigation/control/configs/README.md). Note that you do not need to define all these parameters to evaluate the planning algorithm, but many are relevant.

### 7b. Evaluate the path planner

You can visualize the evaluation of sampled paths on test data using the test script in the planning folder:

```
python navigation/control/planning/test.py <xml_file> --test_data lunar-nav
```

As usual, test_data is used to indicate which dataset should be used for visualization. The xml_file should be the configuration file chosen or created in the previous step. This command will save examples of the regional competency estimates, the score associated with sampled trajectories, the sampled trajectories below the specified threshold, and the selected trajectory for each image in the ID and OOD test sets to a folder called `results/planning/`. 

Alternatively, you can use the example argument to indicate a single example in the OOD test set to be used for evaluation. Doing so will save the single example image to a folder called `results/examples/`. Optionally, you can also specify the initial position of the vehicle using the init_x and init_y arguments and the goal position using the goal_x and goal_y arguments.

## 8) Design Path-Tracking Controller

### 8a. Define the controller parameters

To design the path-tracking control algorithm, you must specify the parameters for the controller in a configuration file like the examples given in `navigation/control/configs/`. If you simply want to run with the default tracking parameters used for evaluation, you can use the config files provided in this folder, but you will need to change the file locations to match your (absolute) file paths. More details on the configurations are provided in the configs [README](https://github.com/sarapohland/parce/tree/main/navigation/control/configs/README.md). Note that you do not need to define all these parameters to evaluate the tracking algorithm, but many are relevant.

### 8b. Evaluate the tracking controller

To evaluate the performance of the path-tracking controller, first bring up the lunar simulation within the lunar-sim environment:

```
roslaunch lunar_gazebo lunar_test.launch world:=train_world
```

You can visualize the selected and realized paths of the vehicle using the test script in the tracking folder:

```
python navigation/control/tracking/test.py <xml_file> --goal_x 10 --goal_y 5
```

The xml_file should be the configuration file chosen or created in the previous step. This command will save examples of the optimal path selected by the path planner and the realized path executed using the path-tracking conroller, along with the error in vehicle position. Optionally, you can specify the goal position using the goal_x and goal_y arguments.

## 9) Run Competency-Aware Controller

### 9a. Define the controller parameters

You can specify the parameters for the competency-aware controller in a configuration file like the examples given in `navigation/control/configs/`. If you simply want to run with the default controller parameters used for evaluation, you can use the config files provided in this folder, but you will need to change the file locations to match your (absolute) file paths. More details on the configurations are provided in the configs [README](https://github.com/sarapohland/parce/tree/main/navigation/control/configs/README.md).

### 9b. Run controller in lunar simulation

To test the controller with the specified parameters in the lunar simulation, first bring up the lunar simulation in the lunar-sim environment:

```
roslaunch lunar_gazebo lunar_test.launch world:=test_world
```

Then from another terminal, run the following command to start the controller:
```
python navigation/control/controller/run_ctrl.py <xml_file>
```

The xml_file should be the same one chosen/created in the previous step. You can specify a CSV file to record results using the record argument and provide a pickle file to store data for visualization using the visualize argument. You can also use the debug flag for debugging

### 9c. Evaluate controller performance with varying levels of competency awareness

The existing configuration files in the `navigation/control/configs/` folder allow you to test the baseline controller with no competency-awareness, the turning-based controller with only overall competency awareness, the turning-based controller with only regional competency awareness, the trajectory-based controller with only regional competency awareness, the turning-based controller with both overall and regional competency awareness, and the trajectory-based controller with both overall and regional competency awareness. To collect data for each of these six controllers in various scenarios, you can set the waypoint_file in the corresponding XML file, along with the time_limit. The currently implemented scenarios are astro1 (with a time limit 75 sec), astro2 (with a time limit of 90 sec), ladder1 (with a time limit of 75 sec), ladder2 (with a time limit of 90 sec), and habitat1 (with a time limit of 90 sec). The list of waypoints corresponding to each of these scenarios is provided in the folder `navigation/control/wypts/`. After selecting the desired scenario, you can run a trial and record results using the run_ctrl script in the controller folder:

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/1-baseline.xml --record results/navigation/1-baseline.csv
```

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/2-overall.xml --record results/navigation/2-overall.csv
```

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/3-regional-turn.xml --record results/navigation/3-regional-turn.csv
```

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/4-regional-traj.xml --record results/navigation/4-regional-traj.csv
```

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/5-both-turn.xml --record results/navigation/5-both-turn.csv
```

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/6-both-traj.xml --record results/navigation/6-both-traj.csv
```

To analyze the results you just collected, you can use the analyze script in the controllers folder:

```
python navigation/control/controller/analyze.py <data_dir>
```

The data directory (data_dir) should be the folder where the CSV file(s) you collected in the previous step are stored. By default, this is `results/navigation/`. This command will print a table with the total number of success, timeouts, and collisions across all trials for each controller type, as well as the average navigation time, path length, linear velocity, angular velocity, linear acceleration, and angular acceleration across all of the trials. You can optionally use the scenario flag to print a table of results for each scenario indepentlty, in addition to the results pulled across all scenarios.

### 9d. Create videos of controllers with varying levels of competency awareness

Again, to collect data for each of the six controllers in various scenarios, you can set the waypoint_file in the corresponding XML file, along with the time_limit (see previous step). After selecting the desired scenario, you can run a trial and record data for visualization using the run_ctrl script in the controller folder:

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/1-baseline.xml --visualize results/visualization/data/1-baseline.p
```

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/2-overall.xml --visualize results/visualization/data/2-overall.p
```

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/3-regional-turn.xml --visualize results/visualization/data/3-regional-turn.p
```

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/4-regional-traj.xml --visualize results/visualization/data/4-regional-traj.p
```

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/5-both-turn.xml --visualize results/visualization/data/5-both-turn.p
```

```
python navigation/control/controller/run_ctrl.py navigation/control/configs/6-both-traj.xml --visualize results/visualization/data/6-both-traj.p
```

You can then create a folder of images summarizing the planning process from these trials using the visualize script in the controller folder:

```
python navigation/control/controller/visualize.py <pickle_file> --save_dir <image_folder>
```

The above command will read the data from the specified pickle file (one of the six created previously) and save images summarizing the planning and control process to the specified folder. After generating these images, you can create a video using the create_video script in the navigation utils folder:

```
python navigation/control/utils/create_video.py <image_folder> --video_file <video_file> --fps 10
```

This script will take the images from the specified image folder (the one created above), create a video with the specified frames-per-second (fps), and save the video to the desired file (mp4 recommended). 
