
### Network
We used the pix2pix network as the details can be seen in the original paper of
conditional image-to-image translation. Our images are resized to 128x128x3 for training.Both generator and discriminator are
optimized by two separate adam optimizers with 2e-4 learning rate.The loss for discriminator
is calculated using log loss meanwhile we calculated the weighted sum of mae and log loss
for the generator's loss.The model is then trained for 40 epochs and 128 batch size.
### Evaluation
The evaluation metric we considered for this task was the basic MSE $$ MSE = {1 \over N} \sum_{i=1}^N (y_i \- f(x_i))^2 $$
We also ranked the top 5 models of all epochs based on this metric.Here are the 5 best
models sorted by the reconstruction loss of their generators after 40 epochs
1. Epoch 24: 0.00978126469
2. Epoch 23: 0.01018550340
3. Epoch 22: 0.01076286565
4. Epoch 17: 0.01082501653
5. Epoch 18: 0.01103080343

![Results of the top-5 best models](https://github.com/Moeed1mdnzh/UStarV1/blob/master/assets/preview.jpg)
<br />
***Results of the top-5 best modelse***
<br />
<br />
By training the model for continuous epochs, the model will fall in mode collapse willynilly.The network and techniques will be improved in the future versions of UStar where
we wish to provide better quality reconstructions, higher resolution images and more
flexible configurations.

