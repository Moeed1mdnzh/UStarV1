### Training
Our images are resized to 128x128x3 for training.Both generator and discriminator are
optimized by two separate adam optimizers with 2e-4 learning rate.The loss for discriminator
is calculated using log loss meanwhile we calculated the weighted sum of mae and log loss
for the generator's loss.The model is then trained for 40 epochs and 128 batch size.
