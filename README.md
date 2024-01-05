# Project Title

## Project Overview

This project focuses on the development and implementation of various autoencoder models using PyTorch. Autoencoders are neural networks used for unsupervised learning tasks, primarily for dimensionality reduction and feature learning. In this repository, we explore different architectures of autoencoders, progressively enhancing their complexity and capabilities.

## Models

### Basic Autoencoder

The basic autoencoder (`Autoencoder`) is a foundational model with a simple architecture, consisting of minimal convolutional layers in the encoder and decoder.

![Basic Autoencoder](rm/auto1_net.png)

here is a sample: 
![Autoencoder sample](rm/epoch_10.png)

### Enhanced Autoencoder

The enhanced autoencoder (`Autoencoder_3`) builds upon the basic version by introducing additional convolutional layers, making it capable of capturing more complex features.

![Enhanced Autoencoder](rm/auto3_net.png)

here is a sample: 
![Autoencoder_3 sample](rm/epoch_20.png)

### Ultimate Autoencoder

The ultimate autoencoder (`Autoencoder_3_Ultimate`) is the most advanced model in the series, featuring a deep architecture with high-capacity layers for handling complex and high-dimensional data.

![Ultimate Autoencoder](rm/auto3_ultimate_net.png)

here is a sample: 
![Autoencoder_3_Ultimate sample](rm/epoch_59.png)

## Thanks
The model was trained using the dataset class and datas of [Paint Torch][1]


[1]: https://github.com/yliess86/PaintsTorch2
