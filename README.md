- The read_images_labels() function accepts a file path to the set of images and a file path to the labels and outputs the set of labels and images
- Assuming the ith label corresponds to the ith image
- The first 8 bytes of the label file indicates some magic number and a size

- Softmax is the function that ensures that the outputs of our model are probabilities
- Applies some mathematical transformation to the inputs; TODO: figure out how softmax actually works and how it generates the probability distribution from the individual values

- The to() method of PyTorch (which is called on a model and accepts a device as input) returns a model which operates on a pointer to data which resides on the device passed as an argument
- For instance, if the device is a GPU (specified by “cuda”), then when to is called, the model’s data is copied into VRAM, and a new model object is created which points to this data, and it uses this pointer to access the data, which represents the parameters of the model

- PyTorch can only perform operations between tensors that are on the same device; this means that if we have some model with its parameters on VRAM, the data which we pass it must also exist on VRAM

- Cross entropy will be our loss function; this loss indicates the difference between the predicted probability and the true probability

- Essentially, we have two probability distributions; the one predicted by the model and the true distribution
- The one predicted by the model is produced after running softmax on the outputs of the NN (the logits)
- The true one is just 1 for the true label and 0 for all other labels

- The loss function in this case should quantify the difference between these two distributions

- Key goal is to harshly punish confidently incorrect answers

Cross entropy loss binary case:
L = -(y _ log(p) + (1 - y) _ log(1-p))

- Here, y is the true label (either 1 or 0)
- If y = 1, a good prediction would be a high value of p, so when p is high loss is low
- If y = 0, a good prediction would be when p is low
- Key intuition is that when the difference is great, the inside result is a negative number with a high magnitude (taking the log of a low number), which gets negated and translates to a high loss

Multi-class case:
L = -sum(y_c \* log(p_c))

- Here, y_c is only 1 if it’s the correct prediction
- When we are on the correct prediction, a high probability is rewarded (close to 1 —> log is close to 0 —> loss is low)
- Even though all other terms are nullified, it makes sense that the higher that incorrect probabilities are, the lower the correct probability is, so the loss is correctly reflected

TODO: understand the different behavior during training and why it's done

- we call model.train() at the start of the training loop; we do this to start dropout (randomly drops (sets to 0) some neurons to prevent overfitting; in overfitting, the model learns the statistical noise of the dataset, so dropout forces the layers to take more or less responsibility for the input by taking a probabilistic approach; still not very clear in my head need to read more), BatchNorm (batch normalization), and LayerNorm

- a batch is just some subset of the training data
- an epoch is one full passing of the training data through the model
- in stochastic gradient descent, model parameters are updated after each batch; in regular GD, the entire train dataset goes through the model prior to the updating of any weights (stochastic comes from the fact that these subsets are randomly chosen), I guess primarily SGD is defined by only using one sample to update the weights every time but mini-batch involves a batch/several samples
- after running backprop on many samples, we end up with a bunch of gradients (each sample produces its own gradient), so we average all of these recommendations to get the actual change that we make to each parameter
- so, in each epoch, we run the train loop which goes through each batch and updates the model weights accordingly, then we run the test loop which evaluates the model's performance on the entire test set

- in the getitem dunder, we need to return pytorch tensors
