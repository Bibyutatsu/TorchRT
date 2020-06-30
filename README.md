## TorchRT
As the name suggests this repo contains codes and functions which can help in understanding how to train your model in Pytorch, convert the model to ONNX and then use TensorRT engine to do a very fast inference on the testing dataset. 
In this project I have used Fashion MNIST dataset for training, validation and inferencing.


## Setup

- Please ensure you have `CUDA=10.2` 
- Install `Conda` if not currently present from [here](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/install/linux.html). 
	- Create a new `env` using `conda create -y -n <env_name> python=3.7`
	- activate the env using `source activate <env_name>`
- Install Pytorch
	- `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`
- Install Pytorch Lightning
	- `conda install pytorch-lightning -c conda-forge`
- Install Onnx
	- `conda install -c conda-forge onnx`
- Install Docker from [here](https://docs.docker.com/engine/install/ubuntu/) if not present. Then pull the TensorRT image
	- `docker pull nvcr.io/nvidia/tensorrt:20.03-py3`

That is everything that you will need for running the codes. Now lets look at the structure of the project folder:
```
TorchRT
│   README.md : We are here
└───Inference : Contains C++ and Python codes for inference along with images
│   └───bin : Contains compiled binaries which you can run
│   └───data : Contains images, models and code to create inference data
│   └───python : Contains Jupyter notebooks
│   └───src : Contains c++ scripts and Makefiles
└───Models    : Contains best performing .onnx and .pth models
└───Research  : Contains codes for training
```

## Training
- For training you can directly use the notebook in `Research/train.ipynb`.
- After your training ends your best model will be saved in `Models/best_model.pth`. (It is recommended that you rename it)
- Now to convert your Pytorch model to ONNX, edit `Research/py2onnx.py` and run
	- `cd Research`
	- `python py2onnx.py`

## Inference
For inference, please follow these steps:
- unzip `Inference/data/fashionmnist/images.rar` and put the `images` folder in `Inference/data/fashionmnist`
- `docker run --gpus all -it -v /path/to/BibhashMitra:/data nvcr.io/nvidia/tensorrt:20.03-py3`
- `cd /data/Inference`
- Now if you want to directly use the precompiled binary
	- `cd bin`
	- `./onnx_fashion_mnist`
- If you want to `make` the cpp files
	- `cd src/onnxFashionMNIST`
	- `make`
	- `cd ../../bin`
	- `./onnx_fashion_mnist`

## Inference Dataset (Optional)
I have already included Inference dataset of 3002 images in `Inference/data/fashionmnist/images` (it is compressed inside `images.rar`). But you can create your own random dataset using `Inference/data/fashionmnist/generate_pgms.py`

# About Me
**Bibhash Chandra Mitra**
Email: bibhashm220896@gmail.com; 
[website](https://bibyutatsu.github.io/Blogs/).