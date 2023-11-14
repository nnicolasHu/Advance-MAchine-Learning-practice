from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.nn as nn
from tqdm import tqdm

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "../data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
writer = SummaryWriter("runs/Hangzhou2/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

latent_dim = 10
rnn = RNN(DIM_INPUT, latent_dim , CLASSES)
enthropy_loss = nn.CrossEntropyLoss()


# Learning rate
eps = 0.005
NB_EPOCH = 150

optim = torch.optim.SGD(params=rnn.parameters(),lr=eps)
optim.zero_grad()


for epoch in tqdm(range(NB_EPOCH)):
    compteur = 0
    for (X,y) in data_train:
        #y_onehot = torch.nn.functional.one_hot(y, num_classes=CLASSES)
        h_all = rnn.forward(X)
        yhat = rnn.decode(h_all[-1])
        loss = enthropy_loss(yhat, y)
        loss.backward()
        with torch.no_grad():
            _ , index = torch.max(nn.Softmax(dim=1)(yhat), 1)
            accuracy = torch.sum(index == y)/y.size(0)

        writer.add_scalar('lossRNN/train', loss, epoch*len(data_train) + compteur)
        writer.add_scalar('accuracy/train', accuracy, epoch*len(data_train) + compteur)


        optim.step()
        optim.zero_grad()
        compteur +=1

    with torch.no_grad():
        loss_test = []
        accuracy_test = []
        for (X,y) in data_test:
            h_all = rnn.forward(X)
            yhat = rnn.decode(h_all[-1])
            loss = enthropy_loss(yhat, y)
    
            _ , index = torch.max(nn.Softmax(dim=1)(yhat), 1)
            accuracy = torch.sum(index == y)/y.size(0)
    
            loss_test.append(loss)
            accuracy_test.append(accuracy)
        
        loss_test = torch.stack(loss_test, dim=0)
        accuracy_test = torch.stack(accuracy_test, dim=0)
    
        writer.add_scalar('lossRNN/test', torch.mean(loss_test), epoch)
        writer.add_scalar('accuracy/test', torch.mean(accuracy_test), epoch)
    
    #print(epoch)