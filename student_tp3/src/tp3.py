import matplotlib.pyplot as plt
from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
# Téléchargement des données

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter(
    "runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(
    1).repeat(1, 3, 1, 1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


savepath = Path("model.pch")

#  TODO:


class MonDataset(Dataset):
    def __init__(self, images, labels):
        self.data = images
        self.labels = labels

    def __getitem__(self, index):
        """ retourne un couple ( exemple , label ) correspondant a l’index"""
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


dataset_train = MonDataset(train_images, train_labels)
dataset_test = MonDataset(test_images, test_labels)

train_loader = DataLoader(dataset_train, shuffle=False, batch_size=30)

# Question 1


def show_image(image_tensor):
    # Convert the PyTorch tensor to a NumPy array
    image_array = image_tensor.numpy()

    # If the image is a vector, reshape it back to its 2D form
    if len(image_array.shape) == 1:
        dim = int(image_array.shape[0] ** 0.5)
        image_array = image_array.reshape((dim, dim))

    # Display the image
    plt.imshow(image_array, cmap='gray')  # Grayscale color map
    plt.axis('off')
    plt.show()


# Testing with different batch sizes
# for batch_size in [10, 20, 30]:
#     train_loader = DataLoader(
#         dataset_train, shuffle=False, batch_size=batch_size)
#     for i, (images, labels) in enumerate(train_loader):
#         show_image(images[i])
#         print(
#             f"Batch {i+1} of size {batch_size}: {images.size()} , Labels: {labels[i]}")
#         if i == 2:  # Stop after printing 3 batches
#             break

# Question 2
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Encoder - just the weight matrix
        self.encoder_weights = nn.Parameter(
            torch.randn(hidden_size, input_size))

        # Bias terms for encoder and decoder
        self.encoder_bias = nn.Parameter(torch.randn(hidden_size))
        self.decoder_bias = nn.Parameter(torch.randn(input_size))

    def forward(self, x):
        # Encoding
        encoded = F.relu(F.linear(x, self.encoder_weights, self.encoder_bias))

        # Decoding - using the transpose of encoder weights
        decoded = torch.sigmoid(
            F.linear(encoded, self.encoder_weights.t(), self.decoder_bias))

        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

# Hyperparameters
learning_rate = 0.01
epochs = 10
batch_size = 64
train_loader = DataLoader(dataset_train, shuffle=False, batch_size=batch_size)

# Initialize the model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(784, 128).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).float().to(device)  # Flatten the images

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Checkpointing
    checkpoint_path = f"autoencoder_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)

# Save the final model
torch.save(model.state_dict(), "autoencoder_final.pth")

# Close the TensorBoard writer
writer.close()