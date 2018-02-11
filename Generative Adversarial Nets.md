---
mathjax: true
---

## Generative Adversarial Nets Notes

created by Kaishen, Feb 9, 2018

#### Easy Understanding

In this part, I want to introduce the GAN to you without any detailed mathematics. 

First of all, GAN, as stated in its name, is a generative model. You may ask, why we need generative model or what it can give us. Informally, generative model has the ability to create **"new samples"** that are similar to the origin sample. 

For example, in normal Machine Learning an Deep Learning scenario, we are often provided some samples and asked make a model to handle them. But what if the task changes to generate some similar samples based on the given sample? I.e. we need more pictures that similar to the CIFAR10. Then GAN can help.

***In short, GAN can generate something new based on the things you feed it.***

How does GAN achieve this? The secret is also inside the name, **"Adversarial"**. GAN has two neural network, generator and discriminator. The generator creates the new (orignal paper call it fake) samples, and the  discriminator try to distinguish the new sample from those trues samples you gave. See the below diagram.

![GAN two nn](./pictures/GAN two nn.png)

The generator and the Discriminator are just like two players, who are competing with each other. This is the reason people call this kind of method Adversarial.

After training, both neural networks are good, then we will use the generator network to generate new images.

#### Questions

1.Why we need to train the $D$ first before we train the $G$ ?

>The reason is that 

![GAN-log(1-x)](./pictures/GAN-log(1-x).jpg)





## References

