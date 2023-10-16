from doctest import OutputChecker
from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
import datetime


## datas
data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

index = int(0.1*datax.size(0))
x_test = datax[:index]
y_test = datay[:index]

x_train = datax[index:]
y_train = datay[index:]

meanx, meany = torch.mean(x_test, 0), torch.mean(y_test, 0)
stdx, stdy = torch.std(x_test), torch.std(y_test)

x_train = (x_train - meanx)/stdx
y_train = (y_train - meany)/stdy
x_test = (x_test - meanx)/stdx
y_test = (y_test - meany)/stdy

eps = 0.05

## NN
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.cost = nn.MSELoss()
    
    def forward(self, x):
        return self.fc3( self.fc2( self.fc1(x) ) )


writer = SummaryWriter("runs/NN/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

hidden_size = int((x_train.size(1) + y_train.size(1))/2)
net = Net(x_train.size(1), hidden_size, y_train.size(1))

optim = torch.optim.SGD(params=net.parameters(),lr=eps)
optim.zero_grad()

for i in range(100):
    yhat = net.forward(x_train)
    mse_train = net.cost(yhat, y_train)

    prediction = net.forward(x_test)
    mse_test = net.cost(prediction, y_test)

    writer.add_scalar('lossNN/train', mse_train, i)
    writer.add_scalar('lossNN/test', mse_test, i)
    mse_train.backward()
        
    optim.step()
    optim.zero_grad()



## containers Sequential
net_countainer = nn.Sequential(
    nn.Linear(x_train.size(1), hidden_size),
    nn.Tanh(),
    nn.Linear(hidden_size, y_train.size(1))
)

optim_countainer = torch.optim.SGD(params=net_countainer.parameters(),lr=eps)
optim_countainer.zero_grad()
criterion = nn.MSELoss()

for i in range(100):
    output = net_countainer.forward(x_train)
    mse_countainer_train = criterion(output, y_train)

    prediction = net_countainer.forward(x_test)
    mse_countainer_test = criterion(prediction, y_test)

    writer.add_scalar('lossNN_countainer/train', mse_countainer_train, i)
    writer.add_scalar('lossNN_countainer/test', mse_countainer_test, i)
    mse_countainer_train.backward()
        
    optim_countainer.step()
    optim_countainer.zero_grad()