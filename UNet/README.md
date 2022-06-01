
### Network
We used the pix2pix network as the details can be seen in the original paper of
conditional image-to-image translation. Our images are resized to 128x128x3 for training.Both generator and discriminator are
optimized by two separate adam optimizers with 2e-4 learning rate.The loss for discriminator
is calculated using log loss meanwhile we calculated the weighted sum of mae and log loss
for the generator's loss.The model is then trained for 40 epochs and 128 batch size.
### Evaluation
The evaluation metric we considered for this task was the basic MSE $$ MSE = {1 \over N \sum_{i=1}^N (y_i \- f(x_i))} $$
