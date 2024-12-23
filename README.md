# Deep Learning for Computer Vision 🔥All From Scratch🔥

## Notice
Big thanks to Michigan Online and Justin Johnson for creating and sharing the fantastic [Deep Learning for Computer Vision (EECS598)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/) course online! This repository is fully licensed under EECS598.

</br>

## 🐣 A1. [K-Nearest Neighbor](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A1/knn.ipynb)
 The K-NN algorithm was used to train the CIFAR-10 dataset from scratch for image classification. Cross-validation was performed to find the optimal hyperparameters, and testing was conducted. As a result, a top-1 accuracy of **33.86%** was achieved.
 
<img width="1076" alt="Screenshot 2024-11-24 at 2 58 47 PM" src="https://github.com/user-attachments/assets/e87950f4-72c7-4a48-90ce-795e6e03730e">

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

### A4-2. [Transformer](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A4/Transformers.ipynb) 
**(Note: This lecture was conducted in 2019, prior to the publication of the Vision Transformer (ViT) paper.)**

 For Transformer training, I will use a toy dataset designed for text-based calculations. Here are a few examples from the dataset:

- Expression: BOS NEGATIVE 30 subtract NEGATIVE 34 EOS, Output: BOS POSITIVE 04 EOS

- Expression: BOS NEGATIVE 34 add NEGATIVE 15 EOS, Output: BOS NEGATIVE 49 EOS

I implemented the Transformer's attention block, multi-head attention block, encoder and decoder blocks, as well as layer normalization from scratch using torch.nn modules and achieved **73.83%** for final model accuracy. (Something wrong with my code. Debug here ☹️)


</br>

## 🐓 A5. Object Detection

### A5-1. [One-Stage Object Detector](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A5/one_stage_detector.ipynb)
 I implemented [FCOS:Fully-Convolutional One-Stage Object Detector](https://arxiv.org/pdf/1904.01355) from scratch as a one-stage object detection model and trained it on the PASCAL VOC 2007 dataset. This dataset is relatively small, containing 20 categories with annotated bounding boxes. Unlike classification tasks, mean Average Precision (mAP) is used as the validation metric for evaluation. (Something wrong with my code. Debug here ☹️)

<img width="1037" alt="Screenshot 2024-11-26 at 12 26 17 AM" src="https://github.com/user-attachments/assets/dce876af-5728-42f6-abce-b40556422cd2">

### A5-2. [Two-Stage Object Detector](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A5/two_stage_detector.ipynb)
 I implemented a two-stage object detector based on [Faster R-CNN](https://arxiv.org/pdf/1506.01497), which comprises two main modules: the Region Proposal Network (RPN) and Fast R-CNN. As with previous section in 5-2, I used the PASCAL VOC 2007 dataset and evaluated performance using mean Average Precision (mAP) as the metric. (Something wrong with my code. Debug here ☹️)


</br>

## 🍗 A6. Generative Models & Visualization

### A6-1. [Variational AutoEncoder (VAE)](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A6/variational_autoencoders.ipynb)

<img width="787" alt="Screenshot 2024-11-26 at 12 37 59 AM" src="https://github.com/user-attachments/assets/22bd47d3-137f-411d-8259-c5e73647dc90">

### A6-2. [Generative Adversarial Networks (GAN)](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A6/generative_adversarial_networks.ipynb)

<img width="1329" alt="Screenshot 2024-11-26 at 12 47 43 AM" src="https://github.com/user-attachments/assets/94cfdb7e-328b-460a-b79d-4fbe4e962b6b">

</br>
</br>

<img width="827" alt="Screenshot 2024-11-26 at 12 53 55 AM" src="https://github.com/user-attachments/assets/fadbf7c1-1119-4680-aec2-234e003a7c77">






