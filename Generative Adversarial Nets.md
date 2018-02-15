---
mathjax: true
---

## Generative Adversarial Nets Notes

created by Kaishen, Feb 9, 2018

#### Easy Understanding

In this part, I want to introduce the GAN to you without any detailed mathematics. 

First of all, GAN, as stated in its name, is a generative model. You may ask, why we need generative model or what it can give us. Informally, generative model has the ability to create **"new samples"** that are similar to the origin samples. 

For example, in normal Machine Learning an Deep Learning scenario, we are often provided some samples and asked make a model to handle them. But what if the task changes to generate some similar samples based on the given sample? I.e. we need more pictures that similar to the CIFAR10. Then GAN can help.

***In short, GAN can generate something new based on the things you feed it.***

How does GAN achieve this? The secret is also inside the name, **"Adversarial"**. GAN has two neural network, generator and discriminator. The generator creates the new (orignal paper call it fake) samples, and the  discriminator try to distinguish the new sample from those trues samples you gave. See the below diagram.

![GAN two nn](./pictures/GAN two nn.png)

The generator and the Discriminator are just like two players, who are competing with each other. This is the reason why people call this kind of method Adversarial.

After training, both neural networks are good, then we will use the generator network to generate new images.

#### Deep Understanding

First of all, we have to thank Ian J. Goodfellow for his great [paper](https://arxiv.org/abs/1406.2661). I acknowledge all of the bellow material are just notes for undertanding Ian's paper.

##### Terms

$p_z$ The probaility distribution of input noise variables $z$, which is also known as "latent variable". The $p_z$ is defined by us, someone also called it prior distribution.

$G(z;\theta_g)$ The differentiable mapping/function maps the latent variable $z$ space to data $x$ space. Simply, it is just $x=G(z)$ . In GAN, this funtion is called **generator**, and is by formed by an neural network. This is also the target network we want to learn through the whole story.

$p_g$ The generator's output distribution, i.e. the probability distributon of $G(z)$ obtained when $z\sim p_z$. Since $x=G(z)$, can think this as probability distribution of **generated/faked** samples.

$D(x;\theta_d)$ The function that maps input $x$ to a single scalar range inside the $[0,1]$, representing the probability that input $x$ came from the true data rather than $p_g$. To be more specific, an $x$ whose $D(x)$ is close to 1, has higher change that this $x$ coming from real data, i.e. probably a real sample. This $D(x)$ function is called **discriminator**. 

$p_{data}$ The true data generating distribution.

***Our ultrimate goal is to make the $p_{data}=p_g$, i.e. learn the true data distribution.***

Usually at this time, they will give out the following object function and tell you to do a minimax game.

<div style="text-align:center">

$\underset{G}{\min}\underset{D}{\max}V(D,G)=\mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z \sim p_z(z)}[\log (1-D(G(z)))]$

</div>

#### Questions

1.Why we need to train the $D$ first before we train the $G$ ?

>The reason is that 

![GAN-log(1-x)](./pictures/GAN-log(1-x).jpg)



#### References

https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/