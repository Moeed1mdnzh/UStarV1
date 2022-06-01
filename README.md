# UStarV1
UStar is a big approach in the field of generative adversarial networks with the goal of being a game changer to astrology.
Our goal is to create simulations in astrology using generative networks.In this case, we
aimed to provide images that resemble to realistic stars which is an Image-to-Image problem
that we try to transform a basic painting to an image of a complex star.We used heavy image
processing techniques to create our data using a single grayscale image of a star.Pix2Pix
network is considered for training in this paper.We rank the model of each epoch using
Mean-Squared-Error and then pick the top-5 best models with the lowest evaluation loss for
our final insights
## Introduction
Every single human being has once tried to draw fancy and imaginary astronomical things in
their life including planets, galaxies, stars and so on, but considering stars, have you ever
wondered how would your paintings actually look like in real life? Would they look like white
dwarfs? Or red giant? Or cosmic dusts? Well, by developing UStarV1 we have found a way
to take these paintings into realistic images of stars.
### Inspiration
The letter U in UStar is inspired by two reasons
1. Letter U being a reference to the network used as our generator(U-Net)
2. Letter U being a reference to word Unreal for having the capability to generate
unreal stars.

### Applications
CNNs(Convolutional Neural Networks) have been a game changer to the variety of image
prediction problems such as semantic and instance segmentation, paired and unpaired
image translation, image reconstruction and so on.In the first version of the UStar
model, we use the UNet network as our cGAN(Conditional Generative Adversarial Network) where we wish to solve the paired Image-To-Image translation of basic paintings to
realistic star images.This model is as well provided with a basic application to demonstrate
the usage of UStarV1

[https://github.com/Moeed1mdnzh/UStar-GUI](https://github.com/Moeed1mdnzh/UStar-GUI)

### Goals
However, the translation of dark stars or pure white stars are not guaranteed to be perfectly
realistic or clear as these colors are rarely seen in between stars.
## Contents
The following URLs represent the explanation to each component of our approach
### [Dataset](https://github.com/Moeed1mdnzh/UStarV1/tree/master/data)
About how the the dataset was created
### [Preprocessing](https://github.com/Moeed1mdnzh/UStarV1/tree/master/utilities)
About the augmentation techniques used to increase data
### [Network](https://github.com/Moeed1mdnzh/UStarV1/tree/master/UNet)
Describes architectures of the following
#### [Generator](https://github.com/Moeed1mdnzh/UStarV1/tree/master/UNet/generator)
Generator's architecture
#### [Discriminator](https://github.com/Moeed1mdnzh/UStarV1/tree/master/UNet/Discriminator)
Discriminator's architecture
## Installation & Train
Open up command prompt or terminal(for mac and linux users) and run the following commands in a directory you desire or if you're a google colab user, follow 
[UStarV1 Colab](https://github.com/Moeed1mdnzh/UStarV1/blob/master/UStarV1_Colab.ipynb)
### Installation
`!git clone https://github.com/Moeed1mdnzh/UStarV1`
<br />
`cd UStarV1`
<br />
**Install the required packages**
<br />
`!python setup.py`
<br />
### Train
**Generate the dataset**
<br />
`!python generate_data.py`
<br />
**Train using the following arguments**
- --VISUALIZE: To wether save the visualizations or not
- --EVALUATE: To wether evaluate the model or not
- --SAVE: To wether save the best model or not
Just a reminder that you can modify the training configurations in `UNet/configs.py`

`!python train.py -v True -e True -s True`
<br />
After training, you can find the result of your evaluation in the main directory in two .jpg and .json formats.Your trained models and visualizations can as well be
found in the **UNet** directory.






