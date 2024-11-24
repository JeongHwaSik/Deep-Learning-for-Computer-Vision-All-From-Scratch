# Deep Learning for Computer Vision 🔥All From Scratch🔥

## Notice
Big thanks to Michigan Online and Justin Johnson for creating and sharing the fantastic [Deep Learning for Computer Vision (EECS598)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/) course online! This repository is fully licensed under EECS598.

</br>

## 🐣 A1. [K-Nearest Neighbor](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A1/knn.ipynb)
 The K-NN algorithm was used to train the CIFAR-10 dataset from scratch for image classification. Cross-validation was performed to find the optimal hyperparameters, and testing was conducted. As a result, a top-1 accuracy of **33.86%** was achieved.

<img width="423" alt="Screenshot 2024-11-24 at 1 34 24 PM" src="https://github.com/user-attachments/assets/85e38861-d83a-4cbe-8ae7-a4f934eb1e77">


<br>


</br>

## 🐦 A2. Linear Classifier

### A2-1. [Single-Layer Linear Classifiers](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A2/linear_classifier.ipynb)
 A single-layer neural network is trained from scratch on the CIFAR-10 dataset for image classification. Two different loss functions, SVM loss and SoftMax loss, are used to compare their performance. SVM classifier achieves 9.06% for validation set while SoftMax classifier achieves **39.69%**.

<img width="970" alt="Screenshot 2024-11-24 at 1 34 09 PM" src="https://github.com/user-attachments/assets/ee534daa-7899-4c92-ab4c-82563b25b045">


### A2-2. [Two Layer Linear Neural Network](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A2/two_layer_net.ipynb)
 A two layer linear neural network is trained from scratch on the CIFAR-10 dataset for image classification. Experiments were conducted with neural networks using different hyper-parameters (hidden dimension, regularization term, learning rate) and found out that the optimal validation performance of **52.32%** was achieved!
 
<img width="505" alt="Screenshot 2024-11-24 at 1 34 35 PM" src="https://github.com/user-attachments/assets/a0c06d42-34f6-4d76-b0ef-9a18e7da0333">

<img width="1006" alt="Screenshot 2024-11-24 at 1 52 07 PM" src="https://github.com/user-attachments/assets/a510d9f5-67f8-4006-a663-80918230efc9">

</br>

## 🐔 A3. Fully Connected Neural Networks & Convolutional Neural Networks

### A3-1. [Fully Connected Networks](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A3/fully_connected_networks.ipynb)
 Implement forward and backward functions for Linear layers, ReLU activation, and DropOut from scratch (NOT using torch.nn modules). I built two layer fully connected layers with ReLU activations and it using different optimization algorithms: SGD, RMSProp, and Adam.

<img width="1079" alt="Screenshot 2024-11-24 at 2 12 21 PM" src="https://github.com/user-attachments/assets/63d3b181-1512-48a6-aa44-80526531da60">

### A3-2. [Convolutional Neural Networks](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A3/convolutional_networks.ipynb)
 Implement forward and backward functions for Convolution layers, MaxPooling, and Batch Normalization from scratch (NOT using torch.nn modules). I built three-layer convolutional networks and each layer consists of Convolution-BatchNorm-ReLU-MaxPool. We add another technique called 'Kaiming Intialization' to stabilize model training at the beginning. Using CIFAR-10 dataset, we achieved **71.9%** top-1 accuracy.

 <img width="397" alt="Screenshot 2024-11-24 at 2 21 14 PM" src="https://github.com/user-attachments/assets/1eb52f6f-f577-4af4-811c-6933e7f57a6e">


</br>

## 🐔 A4. Recurrent Neural Network & Transformer

### A4-1. [RNN & LSTM Image Captioning](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A4/rnn_lstm_captioning.ipynb)
 The COCO Captions dataset includes 80,000 training images and 40,000 validation images, each paired with 5 captions provided by workers on Amazon Mechanical Turk. The figure below illustrates examples from the dataset. For this image captioning task, I implemented vanilla RNN and LSTM models, as they are well-suited for processing sequential text data as input.

<img width="1000" alt="Screenshot 2024-11-24 at 2 35 39 PM" src="https://github.com/user-attachments/assets/1819e24f-3414-4b43-b854-5a1cecedc2dd">
<img width="1000" alt="Screenshot 2024-11-24 at 2 42 54 PM" src="https://github.com/user-attachments/assets/fa16bf20-906b-4b5d-94fa-00459c834c24">
<img width="1000" alt="Screenshot 2024-11-24 at 2 43 01 PM" src="https://github.com/user-attachments/assets/2b9b354d-7acb-479e-ba2d-1813e838037c">






