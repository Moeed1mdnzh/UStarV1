### Preprocessing
Here's the magic where we transformed 813 images to 14616 images(Figure 2) using data
augmentation.The following image processing techniques are used in order to generate the
new samples:
- Mirroring the image
- Changing the lighting of the image
- Image rotation
- Zooming in and out
- And finally applying affine transformations
In the next step of our preprocessing pipeline, the classic normalization for
GANs are applied to our images $$ {img \-127.5 \over 127.5} $$

![Preview of the augmented data](https://github.com/Moeed1mdnzh/UStarV1/blob/master/assets/augment.png)
<br />
***Preview of the augmented data***
<br />
