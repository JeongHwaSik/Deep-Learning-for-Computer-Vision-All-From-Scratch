# Deep Learning for Computer Vision 🔥All From Scratch🔥

## Notice
Big thanks to Michigan Online, Andrej Karpathy, and Justin Johnson for creating and sharing the fantastic [Deep Learning for Computer Vision (EECS598)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/) course online! This repository is fully licensed under EECS598.

<br>
</br>

### A1. [K-NN Classifier](https://github.com/JeongHwaSik/cs231n/blob/main/A1/README.md#a1-k-nearest-neighbor-k-nn-classifier)

### A2. [Linear Classifier]()

### A3. [FC & CNN] (https://github.com/JeongHwaSik/cs231n/tree/main/A3#a3-fc-nn--cnn)

### A4. [RNN & Transformer] (https://github.com/JeongHwaSik/cs231n/tree/main/A4#a4-rnn--transformer)


## A5. Object Detection

### A5-1. One-Stage Object Detector
I implemented [FCOS:Fully-Convolutional One-Stage Object Detector](https://arxiv.org/pdf/1904.01355) from scratch as a one-stage object detection model and trained it on the PASCAL VOC 2007 dataset. 
This dataset is relatively small, containing 20 categories with annotated bounding boxes.
Unlike classification tasks, mean Average Precision (mAP) is used as the validation metric for evaluation.

<img width="1200" alt="Screenshot 2024-12-25 at 11 20 11 PM" src="https://github.com/user-attachments/assets/76d34d83-e2ee-4285-9902-7d9905d68044" />

(Detection sucks... Debug here ☹️)

<img width="1200" alt="Screenshot 2024-12-25 at 11 27 13 PM" src="https://github.com/user-attachments/assets/d4b4854e-5b11-479b-ad56-4d9c11963ffc" />

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A5/one_stage_detector.ipynb) for more details about FCOS implementation!

### A5-2. Two-Stage Object Detector
I implemented a two-stage object detector based on [Faster R-CNN](https://arxiv.org/pdf/1506.01497), which comprises two main modules: the Region Proposal Network (RPN) and Fast R-CNN.
I used FCOS as a backbone instead of Fast R-CNN .
As with previous section in 5-1, I used the PASCAL VOC 2007 dataset and evaluated performance using mean Average Precision (mAP) as the metric. 
 
(Detection sucks... Debug here ☹️)

<img width="1200" alt="Screenshot 2024-12-25 at 11 31 39 PM" src="https://github.com/user-attachments/assets/bf73bd33-8372-48d3-984d-bd6718b5065c" />

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A5/two_stage_detector.ipynb) for more details about Faster R-CNN with FCOS implementation!


<br>
</br>

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







