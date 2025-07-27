# Information Theory Basics

## 1. Entropy

## 2. Cross-Entropy

## 3. Kullback–Leibler (KL) Divergence

</br>

# Variational Inference (VI)

In deep learning, generative models that use a latent space are widely employed across various domains, particularly in Computer Vision for tasks like image generation and in Reinforcement Learning for action planning and decision making.
One of the most well-known frameworks in this space is the [Variational Autoencoder (VAE)](https://arxiv.org/pdf/1312.6114) along with its extension [β-VAE](https://openreview.net/pdf?id=Sy2fzU9gl), which is designed to encourage disentangled representations in the latent space.
At the core of the VAE framework lies a technique called **Variational Inference**, which allows for approximate Bayesian Inference in models with latent variables. 
In this tutorial, we'll derive the objective function used in variational inference, stepping through the key mathematical ideas behind it.

<p align="center">
    <img src="https://github.com/user-attachments/assets/e5179327-8489-48db-8247-4e8733894af5" width="50%"/>
</p>

The goal of variational inference is to approximate the true data distribution $p(x)$ (black curve in the figure above) by learning a model distribution $p_{\theta}(x)$ using data $x$ (the orange dots in the figure above), such that the KL divergence between them is minimized. 
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

Finally, we are left with the term $\log{p_{\theta}(x)}$ which we aim to maximize. However, before we can directly use $\log{p_{\theta}(x)}$, we first need to compute the marginal likelihood $p_{\theta}(x)$ using latent variable $z$ as shown below.

<p align="center">
    <img src="https://github.com/user-attachments/assets/b82a4c18-5fbb-4d36-bba0-a76cd1e46412" width="50%"/>
</p>

$$
p_{\theta}(x) = \sum\limits_{z}p_{\theta}(x|z)p_{\theta}(z)
$$

However, in the equation above, we cannot directly compute the marginal likelihood by integrating over all possible values of $z$ as it is generally intractable. 
To address this, we apply the idea of **importance sampling**, using a tractable approximate posterior $q_{\theta}(z|x)$ to prior $p(z)$ to focus computation on the more relevant regions of $z$-space.


$$
= \sum\limits_{z}p_{\theta}(x|z)p_{\theta}(z)\frac{q_{\theta}(z|x)}{q_{\theta}(z|x)}
$$

$$
= E_{z\sim{q_{\theta}(z|x)}}[p_{\theta}(x|z)\frac{p_{\theta}(z)}{q_{\theta}(z|x)}]
$$

Next, we apply the logarithm to the marginal likelihood $p_{\theta}(x)$ in order to derive an objective that we can optimize, specifically, the log-likelihood $\log{p_{\theta}(x)}$ which serves as our ultimate training goal.

$$
\log{p_{\theta}(x)} = \log{E_{z\sim{q_{\theta}(z|x)}}[p_{\theta}(x|z)\frac{p_{\theta}(z)}{q_{\theta}(z|x)}]}
$$

By using Jensen's Inequality,

$$
\ge E_{z\sim{q_{\theta}(z|x)}}[\log{p_{\theta}(x|z)\frac{p_{\theta}(z)}{q_{\theta}(z|x)}}]
$$

$$
= E_{z\sim{q_{\theta}(z|x)}}[\log{p_{\theta}(x|z)}] - D_{KL}(q_{\theta}(z|x)||p_{\theta}(z))
$$

This expression is known as the **Evidence Lower Bound (ELBO)**. It serves as a tractable lower bound on the true log-likelihood $\log{p(x)}$, which is typically intractable to compute directly.
The ELBO consists of two main terms:

**1. Reconstruction Term:**
   
$$
E_{z\sim{q_{\theta}(z|x)}}[\log{p_{\theta}(x|z)}]
$$

This term encourages the model to accurately reconstruct the input $x$ from the latent variable $z$. If we model $p_{\theta}(x|z)$ as a Gaussian distribution $N(\mu_{x|z}, I)$, this expectation is equivalent to minimizing the Mean Squared Error (MSE) between the original input $x$ and the reconstruction $\hat{x}$.
Alternatively, if $x\in${0,1} (i.e., binary data), $p_{\theta}(x|z)$ is typically modeled as a Bernoulli distribution and the reconstruction term corresponds to the Binary Cross-Entropy (BCE) loss.

**2. Regularization (or Complexity) Term:**

$$
-D_{KL}(q_{\theta}(z|x)||p_{\theta}(z))
$$

This term measures how much the approximate posterior $q_{\theta}(z|x)$ deviates from the prior distribution $p(z)$, which is usually chosen to be a standard normal $N(0, I)$. It acts as a regularizer that encourages the latent space to follow the prior distribution, enabling structured and generalizable representations.

If we set the prior $p_{\theta}(z)$ as standard Noraml distribution $N(0, I)$, and $q_{\theta}(z|x)$ to follow Gaussian distribution $N(\mu_{z|x}, \Sigma_{z|x})$, we can represent it in an closed form as follow:

If we set the prior $p_{\theta}(z)$ to be a standard normal distribution $N(0, I)$, and the model approximate posterior $q_{\theta}(z|x)$ as a Gaussian distribution $N(\mu_{z|x}, \Sigma_{z|x})$, then the KL divergence term in the ELBO can be computed in closed form as follows:

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

where can do backpropagation!

Together, maximizing the ELBO balances accurate reconstruction of the data with maintaining a well-structured latent space.



