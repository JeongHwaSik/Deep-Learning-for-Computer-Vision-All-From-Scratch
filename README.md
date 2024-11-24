# Deep Learning for Computer Vision 🔥All From Scratch🔥

## Notice
Big thanks to Michigan Online and Justin Johnson for creating and sharing the fantastic [Deep Learning for Computer Vision (EECS598)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/) course online! This repository is fully licensed under EECS598.

</br>

## 🐣 A1. [K-Nearest Neighbor](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A1/knn.ipynb)
 The K-NN algorithm was used to train the CIFAR-10 dataset from scratch for image classification. Cross-validation was performed to find the optimal hyperparameters, and testing was conducted. As a result, a top-1 accuracy of **33.86%** was achieved.

#### - Visualization of K-NN image classifier
<img width="500" alt="Screenshot 2024-11-23 at 8 41 48 PM" src="https://github.com/user-attachments/assets/e8333694-3696-423b-9441-2146f1b2a03a">

#### - Cross-validation for optimized hyper-parameter(k)
<img width="500" alt="Screenshot 2024-11-23 at 8 42 20 PM" src="https://github.com/user-attachments/assets/5a8cb103-0def-4d27-b6f7-d072c0a0e4d0">


<br>


</br>

## 🐦 A2. Linear Classifier

### A2-1. [Linear Classifiers](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A2/linear_classifier.ipynb)
 A single-layer neural network is trained from scratch on the CIFAR-10 dataset for image classification. Two different loss functions, SVM loss and SoftMax loss, are used to compare their performance. SVM classifier achieves 9.06% for validation set while SoftMax classifier achieves **39.69%**.

#### - SVM classifier weight visualization
<img width="800" alt="Screenshot 2024-11-24 at 12 41 04 PM" src="https://github.com/user-attachments/assets/c21d769d-e9ec-46d1-9e47-62ee06aa0a7b">

#### - SoftMax classifier weight visualization
<img width="800" alt="Screenshot 2024-11-24 at 12 41 46 PM" src="https://github.com/user-attachments/assets/73e0019b-c247-482d-95cc-15322f2e83ab">


### A2-2. [Two Layer Neural Network](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A2/two_layer_net.ipynb)
 A two layer neural network is trained from scratch on the CIFAR-10 dataset for image classification. Experiments were conducted with neural networks using various hidden dimensions (2, 8, 16, 32), and a validation performance of **52.32%** was achieved when the hidden dimension was set to 128.

 #### - Weight visualization
 <img width="650" alt="Screenshot 2024-11-24 at 12 56 45 PM" src="https://github.com/user-attachments/assets/d753d0a4-0814-45a6-9079-82b8637f14fa">

