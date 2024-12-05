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
Alternatively, the code in this repo can be ran using Docker. To set this up, you will need to build the docker image using the command `docker build --tag quant_nnets .`. This will build a docker image with all the neccesary packages installed to run the scripts contained in the repo. To see that the docker image has built successefully, run the command `docker images` and look for `quant_nnets` under the repository column.

## Running Experiments with Docker
Once the image is built, you can start running the experiments. These experiments are set up so that model training and model compression occur in two separate scripts. Once we have a trained network, that network is saved in the directory `trained_network_weights`. To persist that trained model on your local machine, Docker volumes are used. 

The model training can be ran using the following command:
```
docker run -it --name train_container \
                -v [absolute/path/to/repo]/trained_model_weights:/trained_model_weights \
           quant_nnets python model_training.py -m [model]
```
The model can chosen from 'resnet18', 'resnet34', 'resnet50', 'resnet101'. In addition, other hyperparameters, such as the learning rates tested and batch size can be chosen as well.

Once we have a quantized network, that network is saved in the directory `quantized_models`, which is persisted locally through the use of Docker volumes. The model quantization can be ran using the following command:
```
docker run -it --name train_container \
                -v [absolute/path/to/repo]/trained_model_weights:/trained_model_weights \
                -v [absolute/path/to/repo]/logs:/logs \
                -v [absolute/path/to/repo]/quantized_models:/quantized_models \
           quant_nnets python model_training.py -model [model] -ds 'CIFAR10' -wf [path/to/weights]
```


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
