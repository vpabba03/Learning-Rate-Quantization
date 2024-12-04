# Post-training Quantization for Neural Networks with Provable Gurantees with CIFAR-10

## Overview 
This directory contains code necessary to run a post-training neural-network quantization method GPFQ, that
is based on a greedy path-following mechanism. This directory is a fork of https://github.com/YixuanSeanZhou/Quantized_Neural_Nets. It adds a modified run script to support CIFAR10 better. It changes the incoming model to better fit the data as well as adds a basic training and evaluatation script. This is faclitate easily testing many different models on the CIFAR10 dataset, since it is not as computationally as expensive as doing it on the ImageNet dataset. In addition, we include scripts to generate visualizations of the effects of learning rate hyperparameter changes on the quantization of these models.

    @article{zhang2023post,
      title={Post-training quantization for neural networks with provable guarantees},
      author={Zhang, Jinjie and Zhou, Yixuan and Saab, Rayan},
      journal={SIAM Journal on Mathematics of Data Science},
      volume={5},
      number={2},
      pages={373--399},
      year={2023},
      publisher={SIAM}
    }

## Project Structure and Descriptions
```plaintext
.
├── imgs
│   ├── accuracy_differences # Visualizations of the differences in accuracies based on bits used and learning rate.
│   ├── training_visualizations # Visualizations of training metrics by learning rate used.
│   └── weight_distribution_plots 
├── logs # Contains logs of quantization results and the hyperparameters used to generate the quantized models.
│   ├── init_log.py # Command to initialize the quantization log CSV.
│   └── Quantization_Log.csv # Example of a quantization log
├── quantized_models # Folder containing models post quantization
│   ├── resnet18
│   └── resnet50
├── src
│   ├── model_training_notebooks # Folder containing various notebooks that can be used as an alternative way to run the training scripts.
│   ├── visualization_notebooks # Folder containing various notebooks that can be used as an alternative way to generate the visualizations.
│   ├── data_loaders.py # Helper code to help load datasets for models.
│   ├── main.py # Script to run the quantization script.
│   ├── model_training.py # Script to run the model training script.
│   ├── plot_training.py # Script that plots and saves visualizations of training metrics by learning rate used.
│   ├── quantize_neural_net.py # Contains code to perform the quantization of a neural network.
│   ├── quantized_weight_dist.py # Script that plots and saves visualizations of the distributions between the weights of the original models and quantized models.
│   ├── step_algorithm.py # Contains code for the step algorithm used in quantization
│   └── utils.py # Contains code for various helper functions used in quantization and evaluation
├── trained_model_weights # Folder containing saved model weights from training
├── Dockerfile # Dockerfile to build docker image
├── README.md
└── requirements.txt
```

## Installing Dependencies without Docker
We assume a python version that is greater than `3.8.0` is installed in the user's 
machine. In the root directory of this repo, we provide a `requirements.txt` file for installing the python libraries that will be used in our code. 

To install the necessary dependency, one can first start a virtual environment
by doing the following: 
```
python3 -m venv .venv
source .venv/bin/activate
```
The code above should activate a new python virtual environments.

Then one can make use of the `requirements.txt` by 
```
pip3 install -r requirements.txt
```
This should install all the required dependencies of this project. 

## Using Docker
Alternatively, the code in this repo can be ran using docker. To set this up, you will need to build the docker image

## Using CIFAR-10 dataset
Choose a data directory to store the CIFAR10 dataset in the data loaders python file. This will download the CIFAR-10 dataset to this folder when the script is ran.

## Running Experiments with Docker


## Running Experiments without Docker

The implementation of GPFQ and its sparse mode in our paper is contained in `src/main.py`. 

1. Before running the `main.py` file, navigate to the `logs` directory and run `python init_log.py`. This will prepare a log file `Quantization_Log.csv` which is used to store the results of the experiment. 

2. Open the `src` directory and run `python main.py -h` to check hyperparameters, including the number of bits/batch size used for quantization, the scalar of alphabets, the probability for subsampling in CNNs, and regularizations used for sparse quantization etc.

3. To start the experiment, we provide an example: If we want to quantize the ResNet-18 using ImageNet data with bit = 4, batch_size = 512, scalar = 1.16 and the CIFAR-10 dataset, then we can try this:
```
python main.py -model resnet18 -b 4 -bs 256 -s 1.16 -ds 'CIFAR10'
```
There are other options we can select, see `main.py`

Please open Readme.ipynb for more details. 
