# Deep Learning for Computer Vision 🔥All From Scratch🔥

## Notice
Big thanks to Michigan Online and Justin Johnson for creating and sharing the fantastic [Deep Learning for Computer Vision (EECS598)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/) course online! This repository is fully licensed under EECS598.

<br>
</br>

## A1. K-Nearest Neighbor
 The K-NN algorithm was used to train the CIFAR-10 dataset from scratch for image classification. 
 Cross-validation was performed to find the optimal hyperparameters, and testing was conducted. 
 As a result, a top-1 accuracy of **33.86%** was achieved.
 
<img width="1200" alt="Screenshot 2024-12-25 at 11 42 12 PM" src="https://github.com/user-attachments/assets/7d0c493d-2a18-4674-bd79-e8c1d19b14e3" />

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A1/knn.ipynb) for more details about KNN algorithms.

<br>
</br>

## A2. Linear Classifier

### A2-1. Single-Layer Linear Classifiers
A single-layer neural network is trained from scratch on the CIFAR-10 dataset for image classification.
Two different loss functions, SVM loss and SoftMax loss, are used to compare their performance. 
SVM classifier achieves 9.06% for validation set while SoftMax classifier achieves **39.69%**.

<img width="970" alt="Screenshot 2024-11-24 at 1 34 09 PM" src="https://github.com/user-attachments/assets/ee534daa-7899-4c92-ab4c-82563b25b045">

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A2/linear_classifier.ipynb) for more details about singe linear layer network.

### A2-2. Two Layer Linear Neural Network
A two layer linear neural network is trained from scratch on the CIFAR-10 dataset for image classification. 
Experiments were conducted with neural networks using different hyper-parameters (hidden dimension for below-left figure, regularization term for upper-right figure, learning rate for upper-left figure) and found out that the optimal validation performance of **52.32%** was achieved!
 
<img width="1200" alt="Screenshot 2024-12-26 at 12 08 33 AM" src="https://github.com/user-attachments/assets/b52daa94-8461-489d-ac7d-cc2612cd3499" />

After then, I visualized the weights of the first linear layer (W1) both before and after training. Refer to the figure below.
<img width="1006" alt="Screenshot 2024-11-24 at 1 52 07 PM" src="https://github.com/user-attachments/assets/a510d9f5-67f8-4006-a663-80918230efc9">
See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A2/two_layer_net.ipynb) for more details about two layer linear network.

<br>
</br>

## A3. Fully Connected Neural Networks & Convolutional Neural Networks

### A3-1. [Fully Connected Networks](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A3/fully_connected_networks.ipynb)
 Implement forward and backward functions for Linear layers, ReLU activation, and DropOut from scratch (NOT using torch.nn modules). I built two layer fully connected layers with ReLU activations and it using different optimization algorithms: SGD, RMSProp, and Adam.

<img width="1079" alt="Screenshot 2024-11-24 at 2 12 21 PM" src="https://github.com/user-attachments/assets/63d3b181-1512-48a6-aa44-80526531da60">

### A3-2. [Convolutional Neural Networks](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A3/convolutional_networks.ipynb)
 Implement forward and backward functions for Convolution layers, MaxPooling, and Batch Normalization from scratch (NOT using torch.nn modules). I built three-layer convolutional networks and each layer consists of Convolution-BatchNorm-ReLU-MaxPool. We add another technique called 'Kaiming Intialization' to stabilize model training at the beginning. Using CIFAR-10 dataset, we achieved **71.9%** top-1 accuracy.

 <img width="397" alt="Screenshot 2024-11-24 at 2 21 14 PM" src="https://github.com/user-attachments/assets/1eb52f6f-f577-4af4-811c-6933e7f57a6e">

<br>
</br>

## A4. Recurrent Neural Network & Transformer

### A4-1. RNN & LSTM Image Captioning
 
The COCO Captions dataset includes 80,000 training images and 40,000 validation images, each paired with 5 captions provided by workers on Amazon Mechanical Turk. 
The figure below illustrates examples from the dataset. 
For this image captioning task, I implemented vanilla RNN and LSTM models, as they are well-suited for processing sequential text data as input.

<img width="1200" alt="Screenshot 2024-11-24 at 2 35 39 PM" src="https://github.com/user-attachments/assets/1819e24f-3414-4b43-b854-5a1cecedc2dd">

- RNN: RNN stands for Recurrent Neural Network, designed specifically to handle sequential data. Unlike tasks such as image classification, 
where inputs (images) have a fixed size, language-related tasks like machine translation involve input sequences of varying lengths. 
To address this challenge, RNNs were introduced.

<img width="1200" alt="Screenshot 2024-12-25 at 11 15 21 PM" src="https://github.com/user-attachments/assets/18ab0f85-78f1-4c4a-b4fd-55d21e3349c5" />

- [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf): LSTM stands for Long Short-Term Memory, a variant of RNNs. One of the major limitations of RNNs is the gradient vanishing/exploding problem during training. 
While gradient clipping can help mitigate the exploding gradient issue, resolving the vanishing gradient problem proved far more challenging. 
To overcome this, LSTM architecture was proposed, which improves gradient flow—similar to the effect of residual connections in [ResNet](). 
LSTMs excel at preserving long-term dependencies, making them more effective than standard RNNs for many tasks.
(Note: [GRU](https://www.mt-archive.net/10/SSST-2014-Cho.pdf), or Gated Recurrent Units, is another popular RNN variant similar to LSTM but with a simpler architecture.)

<img width="1200" alt="Screenshot 2024-12-25 at 11 15 33 PM" src="https://github.com/user-attachments/assets/ee02035e-9d13-4c07-8cb1-91698dd0602b" />

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A4/rnn_lstm_captioning.ipynb) for more details about implementation of RNN and LSTM!

### A4-2. Transformer 
**(Note: This lecture was conducted in 2019, prior to the publication of the [Vision Transformer paper](https://arxiv.org/pdf/2010.11929).)**

 For Transformer training, I will use a toy dataset designed for text-based calculations. Here are a few examples from the dataset:

- Expression: BOS NEGATIVE 30 subtract NEGATIVE 34 EOS, Output: BOS POSITIVE 04 EOS

- Expression: BOS NEGATIVE 34 add NEGATIVE 15 EOS, Output: BOS NEGATIVE 49 EOS

I implemented the Transformer's attention block, multi-head attention block, encoder and decoder blocks, as well as layer normalization from scratch using torch.nn modules and achieved **73.83%** for final model accuracy.
See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A4/Transformers.ipynb) for more details about transformer implementation!

<br>
</br>

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

### A6-1. Variational AutoEncoder (VAE)
VAE, which stands for Variational AutoEncoder, is a type of generative model p(x) that incorporates a probabilistic approach into the traditional autoencoder. 
Given an input x, the encoder compresses the data into a latent space z represented as q(z|x), while the decoder reconstructs x from the latent representation z as p(x|z).
Here, I used MNIST dataset to train the VAE. See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A6/variational_autoencoders.ipynb) for more details about VAE and conditional VAE.
<img width="1200" alt="Screenshot 2024-12-25 at 11 36 10 PM" src="https://github.com/user-attachments/assets/3dc7c06f-40aa-43c9-a186-ec7d4b4021af" />

### A6-2. [Generative Adversarial Networks (GAN)](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A6/generative_adversarial_networks.ipynb)

<img width="1600" alt="Screenshot 2024-11-26 at 12 47 43 AM" src="https://github.com/user-attachments/assets/94cfdb7e-328b-460a-b79d-4fbe4e962b6b">

<br>
</br>

<img width="827" alt="Screenshot 2024-11-26 at 12 53 55 AM" src="https://github.com/user-attachments/assets/fadbf7c1-1119-4680-aec2-234e003a7c77">






