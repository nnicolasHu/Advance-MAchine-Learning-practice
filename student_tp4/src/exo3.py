from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch

from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.nn as nn
from tqdm import tqdm

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_DATA = 2 * CLASSES
#Taille du batch
BATCH_SIZE = 32

PATH = "../data/"
torch.autograd.set_detect_anomaly(True)

matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_DATA], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_DATA], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries temporelles
writer = SummaryWriter("runs/Hangzhou3/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

latent_dim = 10
rnn = RNN(DIM_DATA, latent_dim , DIM_DATA)
mse_loss = nn.MSELoss(reduction='sum')


# Learning rate
eps = 0.001
NB_EPOCH = 500

optim = torch.optim.SGD(params=rnn.parameters(),lr=eps)
optim.zero_grad()


for epoch in tqdm(range(NB_EPOCH)):
    compteur = 0
    for (X,y) in data_train:
        input = torch.flatten(X, start_dim=2)
        target = torch.flatten(y, start_dim=2)
        target = torch.permute(target,(1,0,2)) # LENGTH x BATCH_SIZE x DIM_DATA

        h_all = rnn.forward(input)
        yhat = rnn.decode(h_all)

        loss = mse_loss(yhat, target)/(X.size(0)*(LENGTH-1))
        loss.backward()

        writer.add_scalar('lossRNN/train', loss, epoch*len(data_train) + compteur)


        optim.step()
        optim.zero_grad()
        compteur +=1

    with torch.no_grad():
        loss_test = []
        for (X,y) in data_test:
            input = torch.flatten(X, start_dim=2)
            target = torch.flatten(y, start_dim=2)
            target = torch.permute(target,(1,0,2))

            h_all = rnn.forward(input)
            yhat = rnn.decode(h_all)

            loss = mse_loss(yhat, target)/(X.size(0)*(LENGTH-1))
            loss_test.append(loss)
        
        loss_test = torch.stack(loss_test, dim=0)
        writer.add_scalar('lossRNN/test', torch.mean(loss_test), epoch)
    