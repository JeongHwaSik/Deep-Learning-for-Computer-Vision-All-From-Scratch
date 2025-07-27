# Deep Learning for Computer Vision 🔥All From Scratch🔥

## Notice
Big thanks to Michigan Online, Andrej Karpathy, and Justin Johnson for creating and sharing the fantastic [Deep Learning for Computer Vision (EECS598)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/) course online! This repository is fully licensed under EECS598.

<br>
</br>

### A1. [K-NN Classifier](https://github.com/JeongHwaSik/cs231n/blob/main/A1/README.md#a1-k-nearest-neighbor-k-nn-classifier)

### A2. [Linear Classifier]()

### A3. [FC & CNN](https://github.com/JeongHwaSik/cs231n/tree/main/A3#a3-fc-nn--cnn)

### A4. [RNN & Transformer](https://github.com/JeongHwaSik/cs231n/tree/main/A4#a4-rnn--transformer)

### A5. [Object Detection](https://github.com/JeongHwaSik/cs231n/tree/main/A5#a5-object-detection)

## A6. Generative Models & Visualization

### A6-0. Variational Inference (VI)

### A6-1. Variational AutoEncoder (VAE)
VAE, which stands for Variational AutoEncoder, is a type of generative model $p(x)$ that incorporates a probabilistic approach into the traditional autoencoder. 
Given an input $x$, the encoder compresses the data into a latent space $z$ represented as $q(z|x)$, while the decoder reconstructs $x$ from the latent representation $z$ as $p(x|z)$.
Here, I used MNIST dataset to train the VAE. 
(Conditional VAE is almost the same as VAE except that it has conditional input $x$ given $y$.)

<img width="1200" alt="Screenshot 2024-12-25 at 11 36 10 PM" src="https://github.com/user-attachments/assets/3dc7c06f-40aa-43c9-a186-ec7d4b4021af" />

 See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A6/variational_autoencoders.ipynb) for more details about VAE and conditional VAE.

### A6-2. Generative Adversarial Networks (GAN)

GAN, which stands for Generative Adversarial Network, is a type of generative model $p(x)$ that employs two neural networks in a competitive framework: a generator and a discriminator. 
The generator creates synthetic data $G(z)$ from a latent space $z$, while the discriminator attempts to distinguish between real data $x$ and generated data $G(z)$. 
Both networks are trained simultaneously, improving each other's performance iteratively.

Here, I implemented two types of GAN: "Deeply Convolutional GAN" and "Fully Connected GAN".
Figure below shows generated images for DCGAN with latent interpolation.

<p align="center">
<img width="600" alt="Screenshot 2024-12-27 at 10 16 24 PM" src="https://github.com/user-attachments/assets/73bb20b8-ab47-4003-b256-e5da8ded35ea" />
</p>

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A6/generative_adversarial_networks.ipynb) for more details about GAN!







