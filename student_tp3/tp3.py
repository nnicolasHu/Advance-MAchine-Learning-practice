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
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


savepath = Path("model.pch")

#  TODO: 
class MonDataset(Dataset):
    def __init__(self, images, labels):
        self.data = images/255
        self.labels = labels
    
    def __getitem__(self, index):
        """ retourne un couple ( exemple , label ) correspondant a l’index"""
        return self.data[index], self.labels[index]
        
    
    def __len__(self):
        return len(self.labels)

dataset_train = MonDataset(train_images, train_labels)
dataset_test = MonDataset(test_images, test_labels)

train_loader = DataLoader(dataset_train, shuffle=False, batch_size=30)


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu1 = nn.ReLU()
        self.sig1 = nn.Sigmoid()


    def forward(self, x):
        encoded = self.relu1(self.fc1(x))
        decoded = self.sig1(encoded @ self.fc1.weight.T)
        return decoded



