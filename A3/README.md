# A3. FC-NN & CNN

## A3-1. Fully Connected Neural Networks
I implemented forward and backward functions for Linear layers, ReLU activation, and DropOut from scratch without using `nn.Linear.forward()` and `loss.backward()`❗️. 
Then, I built two fully connected linear layers with ReLU non-linearity using different optimization algorithms: SGD, RMSProp, and Adam.

<img width="1200" alt="Screenshot 2024-12-27 at 12 37 12 AM" src="https://github.com/user-attachments/assets/8fea0929-9cbc-453c-b442-fc5b2b91bc77" />

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A3/fully_connected_networks.ipynb) for more details about two layer linear networks and related experiments.

## A3-2. Convolutional Neural Networks
I implemented forward and backward functions for Convolution layers, MaxPooling, and Batch Normalization from scratch without using `nn.Conv2d.forward()` and `loss.backward()`❗️.
(I used three consecutive for-loops to implement forward and backward passes for convolution layers as convolution operates over dimensions of batch size, kernel size, and width & height.)
After then, I built three-layer convolutional networks and each layer consists of Convolution-BatchNorm-ReLU-MaxPool blocks. 
I add another technique called [Kaiming Initialization](https://arxiv.org/pdf/1502.01852v1) to stabilize model training at the beginning. 
Using CIFAR-10 dataset, I achieved **71.9%** top-1 accuracy.
The figure below shows the trained image of the first convolution kernel, which is entirely different from the weights of the linear layer shown in A2. 
It resembles edges or one-dimensional shapes.

<p align="center">
<img width="600" alt="Screenshot 2024-12-27 at 12 25 54 AM" src="https://github.com/user-attachments/assets/655c9535-5961-4713-a7a3-e6c7215b61fe" />
</p>

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A3/convolutional_networks.ipynb) for more details about convolution, maxpool, and batchnorm operators.
