---
layout: post
title: "Using Auto-Encoders to denoise domain specific images"
date: 2018-09-20
description: Introduction to Image denoising using PyTorch! and MNIST# Add post description (optional)
img:   # Add image post (optional)
url: blog/denoising-auto-encoders
---
# Using Auto-Encoders to denoise domain specific images

---

Auto encoders are neural network architectures that are used to replicate the input data. They have symmetrical layers that are trained on the input data with the target being the input data itself. This allows us to train useful representations of the input data. Few of the applications of auto-encoders are:

- Dimensionality reduction

- Dataset Augmentation

- Latent representation of the data

- Denoising corrupted images

  

---

In this post, I will demonstrate the use of auto-encoders for image denoising. The reason behind the use of auto-encoders for denoising is the extreme effectiveness of auto-encoders in replicating the input data.

First, let us build a simple auto-encoder to recontruct the input data. We use MNIST with a batch size of 64 minimized over mean squared error using RMSprop optimizer. The model is trained for 30 epochs.

The following image shows the output of a simple auto-ecnoder :
![autoencoder-op image]({{site.baseurl}}/assets/img/denoising_autoencoders/autoencoder_op.png)
  

---
Now, in order to add the denoising capability to simple auto-encoders, we apply a transformation to the input data to mimic the corruption that is expected in future data. The transformed input is used to train the data and the loss is computed between the output of the final layer and the untransformed(uncorrupted) data. This helps the network to reconstruct the actual image from a corrupted representation of the image.

The following image describes the architecture of a Denoising auto-encoder :

![Network image]({{site.baseurl}}/assets/img/denoising_autoencoders/network.png)
  

The denoising auto-encoder uses similar hidden layers as of the simple auto-encoder described above. It has a single hidden layer with 300 nodes. This hidden layer is called the bottle-neck of the network and provides a representation of the input.

  For the noise generation, I used NumPy to randomly sample values from a normal distribution with a mean 0 and standard deviation of 1. The sampled values are added to the input image. The obtained array is clipped between 0 and 1. The clipping is necesary because the pixel values belong to that range. The following code is use to corrupt the image :

```python

def corrupt_image(image_data, corruption_factor=0.5):

corrupted_image = image_data + corruption_factor * np.random.normal(loc=0.0, scale=1.0, size=image_data.shape)

corrupted_image = np.clip(corrupted_image, 0., 1.)

return corrupted_image

```

The following image shows the output of the denosing auto-encoder:

![denoising-op image]({{site.baseurl}}/assets/img/denoising_autoencoders/denoising_autoencoder_op.png)

> The __*corruption\_factor*__ decides the amount of noise added to the input. We can tweak with this parameter to train models accordingly.

Further, I have trained the network by using random corruption factor. I used NumPy to randomly sample floats between 0 and 1 from a continous uniform distribution.  The results are shown below :
![Noise random image]({{site.baseurl}}/assets/img/denoising_autoencoders/denoising_autoencoder_randomnoise_op.png)
>  - As  I have mentioned in the title, these denoising methods are trained on previous data. So, they are domain or data specific in their functionality. In simple terms, the network trained in this article is good only for images explicitly containing digits.
>  - Also, using convolutional layers helps in building better systems. 

The code for this article is available on my [Github](https://github.com/amdsrinivas/Blog-Codes). 
