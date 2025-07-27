# A2. Linear Classifier

## A2-1. Single Layer Linear Neural Network
A single-layer neural network is trained from scratch on the CIFAR-10 dataset for image classification.
Here, I avoided using `nn.Linear.forward()` and `loss.backward()`. 
Instead, I implemented the forward pass of the linear layer and the backward pass (manually calculating gradients using the chain rule) entirely from scratch❗️
Two different loss functions, SVM loss and SoftMax loss, are used to compare their performance and they are also implemented from the bottom without using PyTorch modules.
SVM classifier achieves **38.99%** for validation set while SoftMax classifier achieves **39.69%**.
The figure below visualizes a learned weights of the linear layer. As you can see, the weights attempt to mimic the original object but a little bit blurry.

<img width="1200" alt="Screenshot 2024-12-26 at 11 44 31 AM" src="https://github.com/user-attachments/assets/19ebdf86-40db-492b-960f-b7fdc0c50aa9" />

See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A2/linear_classifier.ipynb) for more details about singe linear layer network.

## A2-2. Two Layer Linear Neural Network
A two layer linear neural network is trained from scratch on the CIFAR-10 dataset for image classification.
As I mentioned in A2-1, I implemented the forward and backward passes of two linear layers all from scratch without using `nn.Linear.forward()` and `loss.backward()`❗️
Experiments were conducted with neural networks using different hyper-parameters (hidden dimension for below-left figure, regularization term for upper-right figure, learning rate for upper-left figure) and found out that the optimal validation performance of **52.32%** was achieved!
 
<img width="1200" alt="Screenshot 2024-12-26 at 12 08 33 AM" src="https://github.com/user-attachments/assets/b52daa94-8461-489d-ac7d-cc2612cd3499" />

After then, I visualized the weights of the first linear layer (W1) both before and after training. Refer to the figure below. 
Similar to the learned weights figure in A2-1, the weights here also attempt to mimic the original object but with greater clarity.

<img width="1200" alt="Screenshot 2024-12-26 at 12 10 22 PM" src="https://github.com/user-attachments/assets/1181e007-3886-464b-aef1-04e87a80b13b" />


See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A2/two_layer_net.ipynb) for more details about two layer linear neural network.
