# Information Theory Basics

## 1. Entropy

## 2. Cross-Entropy

## 3. Kullback–Leibler (KL) Divergence

</br>

# Variational Inference (VI)

In deep learning, generative models that use a latent space are widely employed across various domains, particularly in Computer Vision for tasks like image generation and in Reinforcement Learning for action planning and decision making.
One of the most well-known frameworks in this space is the [Variational Autoencoder (VAE)](https://arxiv.org/pdf/1312.6114) along with its extension, the [β-VAE](https://openreview.net/pdf?id=Sy2fzU9gl), which is designed to encourage disentangled representations in the latent space.
At the core of the VAE framework lies a technique called **Variational Inference**, which allows for approximate Bayesian Inference in models with latent variables. 
In this tutorial, I'll derive the objective function used in variational inference, stepping through the key mathematical ideas behind it.

The goal of variational inference is to approximate the true data distribution $p(x)$ by learning a model distribution $p_{\theta}(x)$ using data $x$, such that the KL divergence between them is minimized. 
In other words, we want our model to generate data that is as close as possible to the real distribution.

$$
minimize \ \ \ D_{KL}(p(x)||p_{\theta}(x))
$$

$$
= E_{x\sim{p(x)}}[\log{\frac{p(x)}{p_{\theta}(x)}}]
$$

$$
= \sum\limits_{x}p(x)\log{\frac{p(x)}{p_{\theta}(x)}}
$$

$$
= \sum\limits_{x}p(x)\log{p(x)} - \sum\limits_{x}p(x)\log{p_{\theta}(x)}
$$

The first term $\sum{p(x)}\log{p(x)}$ is constant with respect to the optimization parameter $\theta$ so it can be ignored during optimization. 
As a result, we focus on optimizing the remaining terms that depend on $\theta$.

$$
maximize \ \ \ \sum\limits_{x}p(x)\log{p_{\theta}(x)}
$$

$$
\approx \frac{1}{N}\sum\limits_{x}\log{p_{\theta}(x)}
$$

Finally, we are left with the term $\log{p_{\theta}(x)}$ which we aim to maximize. However, before we can directly use $\log{p_{\theta}(x)}$, we first need to compute the marginal likelihood $p_{\theta}(x)$.

$$
p_{\theta}(x) = \sum\limits_{z}p_{\theta}(x|z)p_{\theta}(z)
$$

However, in the equation above, we cannot directly compute the marginal likelihood by integrating over all possible values of $z$ as it is generally intractable. 
To address this, we apply the idea of **importance sampling**, using a tractable approximate posterior $q_{\theta}(z|s)$ to focus computation on the more relevant regions of $z$-space.


$$
= p_{\theta}(x) = \sum\limits_{z}p_{\theta}(x|z)p_{\theta}(z)\frac{q_{\theta}(z|x)}{q_{\theta}(z|x)}
$$

$$
= E_{z\sim{q_{\theta}(z|x)}}[p_{\theta}(x|z)\frac{p_{\theta}(z)}{q_{\theta}(z|x)}]
$$

Then we're going to add $\log$ to compute our goal.

$$
\log{p_{\theta}(x)} 
$$

$$
= \log{E_{z\sim{q_{\theta}(z|x)}}[p_{\theta}(x|z)\frac{p_{\theta}(z)}{q_{\theta}(z|x)}]}
$$

By using Jensen's Inequality,

$$
\ge E_{z\sim{q_{\theta}(z|x)}}[\log{p_{\theta}(x|z)\frac{p_{\theta}(z)}{q_{\theta}(z|x)}}]
$$

$$
= E_{z\sim{q_{\theta}(z|x)}}[\log{p_{\theta}(x|z)}] - D_{KL}(q_{\theta}(z|x)||p_{\theta}(z))
$$

This is Evidence Lower Bound (ELBO). The first term $E_{z\sim{q_{\theta}(z|x)}}[\log{p_{\theta}(x|z)}]$ is called "reconstruction" term and the second term $D_{KL}(q_{\theta}(z|x)||p_{\theta}(z))$ is called "complexity" (or "regularization") term. 
For the first term, if we set $p_{\theta}(x|z)$ as Gaussian distribution $N(\mu_{x|z}, I)$, then the reconstruction term is same as calculating Mean Squared Error (MSE) $||x-\hat{x}||^2_2$. 
On the other hand, if $x\in{0,1}$, which is binary, then the reconstruction term is calculated as Binary Cross-Entropy term.


If we set the prior $p_{\theta}(z)$ as standard Noraml distribution $N(0, I)$, and $q_{\theta}(z|x)$ to follow Gaussian distribution $N(\mu_{z|x}, \Sigma_{z|x})$, we can represent it in an closed form as follow:

$$
D_{KL}(q_{\theta}(z|x)||p_{\theta}(z)) 
$$

$$
= \int_z{q_{\theta}(z|x)\log{\frac{p_{\theta}(z)}{q_{\theta}(z|x)}}}dz
$$

$$
= \int_z{N(z; \mu_{z|x}, \Sigma_{z|x})\log{\frac{N(z; 0, I)}{N(z; \mu_{z|x}, \Sigma_{z|x})}}}dz
$$

$$
= \sum\limits_{j=1}^J(1 + \log((\sum_{z|x})^2_{j}) - (\mu_{z|x})^2_{j} - (\Sigma_{z|x})^2_{j})
$$

where can can use backpropagation!



