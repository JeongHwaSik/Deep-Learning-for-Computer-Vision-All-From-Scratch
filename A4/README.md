# A4. RNN & Transformer

## A4-1. RNN & LSTM Image Captioning
 
The COCO Captions dataset includes 80,000 training images and 40,000 validation images, each paired with 5 captions provided by workers on Amazon Mechanical Turk. 
The figure below illustrates examples from the dataset. 
For this image captioning task, I implemented vanilla RNN and LSTM models, as they are well-suited for processing sequential text data as input.
I implemented those models from scratch using only `torch.nn` modules without using built-in `nn.RNN()` and `nn.LSTM()`❗️

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

## A4-2. Transformer 
**(Note: This lecture was conducted in 2019, prior to the publication of the [Vision Transformer paper](https://arxiv.org/pdf/2010.11929).)**

Based on [Attention is All You Need](https://arxiv.org/pdf/1706.03762) paper, I implemented the Transformer's Self-Attention, Multi-head Attention, Encoder and Decoder blocks, as well as Layer Normalization from scratch using `torch.nn` modules (without using `nn.MultiheadAttention()`, `nn.LayerNorm()`)❗️
I used a simple toy dataset designed for text-based calculations. Here are a few examples from the dataset:

- Expression: BOS NEGATIVE 30 subtract NEGATIVE 34 EOS, Output: BOS POSITIVE 04 EOS

- Expression: BOS NEGATIVE 34 add NEGATIVE 15 EOS, Output: BOS NEGATIVE 49 EOS

By training transformer seq2seq models with those text-based calculation dataset, I could get **69.92%** accuracy for final model accuracy.
See [🔥here🔥](https://github.com/JeongHwaSik/Deep-Learning-for-Computer-Vision-All-From-Scratch/blob/main/A4/Transformers.ipynb) for more details about transformer implementation!

<br>
</br>
