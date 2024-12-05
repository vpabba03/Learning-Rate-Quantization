# Impact of Training Learning Rates in Quantized Models with CIFAR-10

## Overview 
This directory contains code necessary to run a post-training neural-network quantization method GPFQ, that is based on a greedy path-following mechanism. This directory is a fork of https://github.com/YixuanSeanZhou/Quantized_Neural_Nets. It adds a modified run script to support CIFAR10 better. It changes the incoming model to better fit the data as well as adds a basic training and evaluation script. This is to facilitate easily testing many different models on the CIFAR10 dataset, since it is not as computationally expensive as doing it on the ImageNet dataset. In addition, we include scripts to generate visualizations to assist in the analysis of learning rate hyperparameters on quantized models.

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
├── imgs                                # Directory for visualizations
│   ├── accuracy_differences           # Plots of accuracy differences based on bits used and learning rate
│   ├── training_visualizations        # Visualizations of training metrics by learning rate
│   └── weight_distribution_plots      # Comparisons of pre-quantization and quantized weight distributions
├── logs                                # Logs of quantization results and hyperparameters
│   ├── init_log.py                    # Script to initialize the quantization log CSV
│   └── Quantization_Log.csv           # Example quantization log file
├── quantized_models                    # Contains models after quantization
│   ├── resnet18                       # Quantized ResNet-18 models
│   └── resnet50                       # Quantized ResNet-50 models
├── src                                 # Source code and utilities
│   ├── model_training_notebooks       # Jupyter notebooks for training models
│   ├── visualization_notebooks        # Jupyter notebooks for generating visualizations
│   ├── data_loaders.py                # Helper functions for loading datasets
│   ├── main.py                        # Main script for running quantization experiments
│   ├── model_training.py              # Script for training neural network models
│   ├── plot_training.py               # Script for visualizing training metrics
│   ├── quantize_neural_net.py         # Core logic for neural network quantization
│   ├── quantized_weight_dist.py       # Script for visualizing weight distributions (quantized vs. original)
│   ├── step_algorithm.py              # Implements the step algorithm for quantization
│   └── utils.py                       # Helper functions for various tasks (e.g., evaluation, preprocessing)
├── trained_model_weights               # Trained model weights
├── Dockerfile                          # Docker configuration file for building an image
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies for the project

```
## Getting Started
### **Initialize the Quantization Log**  
Before running any experiments, initialize the quantization log using:

```bash
python logs/init_log.py
```
## Using Docker
Alternatively, the code in this repo can be run using Docker. To set this up, you will need to build the Docker image using the command `docker build --tag quant_nnets .`. This will build a Docker image with all the necessary packages installed to run the scripts contained in the repo. To see that the Docker image has built successfully, run the command `docker images` and look for `quant_nnets` under the repository column. Once the image is built, you can start running the experiments. These experiments are set up so that model training and model compression occur in two separate scripts. Each type of visualization also can be run using its own individual script. 

### Training Networks with Docker
Once we have a trained network, that network is saved in the directory `trained_model_weights`. To persist that trained model on your local machine, Docker volumes are used. The model training can be run using the following command:
```
docker run -it --name train_container \
                -v [absolute/path/to/repo]/trained_model_weights:/trained_model_weights \
           quant_nnets python model_training.py -m [model]
```
The model can be chosen from 'resnet18', 'resnet34', 'resnet50', 'resnet101'. In addition, other hyperparameters, such as the learning rates tested and batch size, can be chosen as well.

### Quantizing Networks with Docker
Once we have a quantized network, that network is saved in the directory `quantized_models`, which is persisted locally through the use of Docker volumes. The model quantization can be run using the following command:
```
docker run -it --name quantization_container \
                -v [absolute/path/to/repo]/trained_model_weights:/trained_model_weights \
                -v [absolute/path/to/repo]/logs:/logs \
                -v [absolute/path/to/repo]/quantized_models:/quantized_models \
           quant_nnets python main.py -model [model] -ds 'CIFAR10' -wf [path/to/weights]
```

### Generating Visualizations with Docker
There are three different visualization types that are generated. These all use Docker volumes to persist on your local machine.

To generate visualizations of training metrics, run the following command:
```
docker run -it --name training_viz_container \
                -v [absolute/path/to/repo]/trained_model_weights:/trained_model_weights \
                -v [absolute/path/to/repo]/imgs:/imgs \
           quant_nnets python plot_training.py --model [model] --output_dir [path/to/output/dir]
```

**NOTE**: The model argument used in this command should be an integer corresponding to one of the ResNet model sizes (18 for resnet18, 34 for resnet34, etc).

To generate visualizations of the distributions of the pre-quantization and quantized weights, run the following command:
```
docker run -it --name weight_dist_viz_container \
                -v [absolute/path/to/repo]/trained_model_weights:/trained_model_weights \
                -v [absolute/path/to/repo]/imgs:/imgs \
                -v [absolute/path/to/repo]/quantized_models:/quantized_models \
           quant_nnets python quantized_weight_dist.py --quantized_dir [path/to/quantized/models] --original_dir [path/to/unquantized/models] --output_dir [path/to/output/dir]
```

To generate visualizations of the accuracy differences between models based on bits and learning rate, run the following command:
```
docker run -it --name acc_dif_viz_container \
                -v [absolute/path/to/repo]/logs:/logs \
                -v [absolute/path/to/repo]/imgs:/imgs \
           quant_nnets python acc_diffs.py --model [model] --experiments-csv [path/to/experiment/csv]
```

## Installing Dependencies without Docker
We assume a Python version that is greater than `3.8.0` is installed on the user's machine. In the root directory of this repo, we provide a `requirements.txt` file for installing the Python libraries that will be used in our code. 

To install the necessary dependencies, one can first start a virtual environment by doing the following: 
```
python3 -m venv .venv
source .venv/bin/activate
```
The code above should activate a new Python virtual environment.

Then one can make use of the `requirements.txt` by 
```
pip3 install -r requirements.txt
```
This should install all the required dependencies of this project. 

## Running Experiments without Docker
Running experiments once the environment is set up should be similar to running the Docker commands. To do so, navigate to the `src` folder and simply run the commands using `python [script]` followed by any necessary arguments. These can be found for any given script by running `python [script] -h`. In addition, Jupyter notebooks for the training and visualizations are included for further fidelity if needed.
