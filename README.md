# distributed-model-training-ray

Description: This project demonstrates how to train a PyTorch neural network model in a distributed manner using Ray. The project implements a distributed training pipeline that scales across multiple workers, optimizing communication and synchronization through Ray's TorchTrainer. This setup allows efficient training of large models on multiple machines or GPUs, leveraging advanced techniques such as all-reduce for gradient synchronization.

Key Features:

Distributed training using Ray's TorchTrainer with multiple workers.

Neural network training on the FashionMNIST dataset using PyTorch.

Data parallelism, worker synchronization, and model gradient aggregation using the all-reduce algorithm.

Configurable to run on both CPU and GPU environments.

Demonstrates integration of data loaders and model preparation using ray.train.torch.


Technologies Used:

Ray for distributed training
PyTorch for model building and training
TorchTrainer for scaling and synchronization
FashionMNIST dataset for image classification
