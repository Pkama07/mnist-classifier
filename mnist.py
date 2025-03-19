# we are working with four things here
# the training images
# the training labels
# the test images
# and the test labels

# we should have one function which accepts the filepath to the images and labels and returns the list

import struct
from array import array
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset


BATCH_SIZE = 32
mps_device = torch.device("mps")


class MNISTDataset(Dataset):

    def __init__(
        self,
        images_filepath,
        labels_filepath,
    ):
        self.images_filepath = images_filepath
        self.labels_filepath = labels_filepath
        self.set_data()

    def set_data(self):
        self.images, self.labels = self.read_data(
            self.images_filepath, self.labels_filepath
        )

    def read_data(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array(
                "B", file.read()
            )  # image data is the section of the file AFTER the magic, size, rows, and cols have been read; only contains data for the images
            # so each image is just described by a sequence of 784 numbers between 0 and 255 representing the brightness of that pixel
            # so, the NN that we create should have 784 input nodes
        images = []
        for i in range(size):
            images.append(
                [0] * rows * cols
            )  # creates a 1D array with 28x28 slots for the pixels of the images
        for i in range(size):
            img = np.array(
                image_data[i * rows * cols : (i + 1) * rows * cols]
            )  # read the section of the data
            img = img.reshape(
                28, 28
            )  # change the shape of the array without changing its data
            images[i][:] = img

        # so now
        # images is a list of 60,000 28x28 arrays which indicate the pixel intensities of each drawing
        # labels is a list of 60,000 labels for these images
        return images, labels

    def __getitem__(self, idx):
        # we need to move the tensors to the mps device
        image_tensor = (
            torch.FloatTensor(np.array(self.images[idx])).view(28 * 28).to(mps_device)
        )
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.int).to(mps_device)
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.images)

    def visualize_mnist_digit(self, data):
        # Convert to numpy array if it's not already
        data = np.array(data)

        # If the data is flattened (1D with 784 elements), reshape it
        if data.ndim == 1 and len(data) == 784:
            data = data.reshape(28, 28)

        # Check if the shape is correct
        if data.shape != (28, 28):
            raise ValueError(
                "Input data must be 28x28 or a flattened 784 element array"
            )

        # Create a figure with a specific size
        plt.figure(figsize=(5, 5))

        # Display the image with a grayscale colormap
        # MNIST digits have values from 0 (white) to 255 or 1 (black)
        plt.imshow(data, cmap="gray")

        # Remove axes for cleaner visualization
        plt.axis("off")

        # Add a title
        plt.title("MNIST Digit")

        # Show the plot
        plt.tight_layout()
        plt.show()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = (
            nn.Flatten()
        )  # returns an instance of the Flatten class which implements __call__ so we can pass input
        self.linear_relu_stack = nn.Sequential(  # implements a neural network; each argument passed represents another layer of the network
            # the weights are initialized through He initialization
            # all biases are initialized to 0
            # so the first layer takes 784 values as input
            nn.Linear(
                28 * 28, 512
            ),  # linear layer applies a linear transformation to the input data (multiplies by the weights and adds the bias)
            nn.ReLU(),  # non-linear function introduces non-linearity (wtf does this actually mean)
            # this particular transformation (ReLU) outputs 0 if the value is negative, and the value if it is positive (max(0, x))
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # is a list of tensors (makes sense)
        logits = self.linear_relu_stack(x)
        return logits  # returns the raw logits; softmax needs to be applied to generate the probabilities


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        # this actually computes the prediction and loss for many samples (all of the samples in the batch)
        pred = model(X)
        # average loss across all the samples
        loss = loss_fn(pred, y)

        # Backpropagation
        # zero the gradients so they don't accumulate between batches
        optimizer.zero_grad()
        # computes the gradient for all of the samples
        loss.backward()
        # takes the average step
        # how does this actually work? How does it change the values of the parameters in the network? how does it know the gradients?
        optimizer.step()

        # so loss.backward calculates all of the gradients

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            # single avg loss value for the whole batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    input_path = "./input"
    training_images_filepath = join(
        input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
    )
    training_labels_filepath = join(
        input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
    )
    test_images_filepath = join(
        input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    )
    test_labels_filepath = join(
        input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
    )

    train_set = MNISTDataset(
        training_images_filepath,
        training_labels_filepath,
    )

    test_set = MNISTDataset(
        test_images_filepath,
        test_labels_filepath,
    )

    model = NeuralNetwork().to(mps_device)

    # how do we run backprop on this model? We need to define a loss function
    criterion = nn.CrossEntropyLoss()  # loss criterion is cross entropy loss
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    EPOCHS = 10
    for i in range(EPOCHS):
        print(f"Running epoch {i}...")
        train_loop(train_loader, model, criterion, optimizer)
        test_loop(test_loader, model, criterion)
