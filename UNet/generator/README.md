We used U-Net for our image generation task which has been one of the best networks for
supervised medical tasks such as tumor segmentation.

The details are explained in the original paper more specific and in-depth.The generator
consists of multiple convolutional blocks gathering featues and reducing the spatial
dimensions to a latent size followed by multiple deconvolutions that include skips connected
to the previous convolutions.
