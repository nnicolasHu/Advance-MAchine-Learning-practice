import torch
from torch.utils.tensorboard import SummaryWriter
# Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
import datetime


data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax, dtype=torch.float)
datay = torch.tensor(datay, dtype=torch.float).reshape(-1, 1)

# meanx, meany = torch.mean(datax, 0), torch.mean(datay, 0)
# stdx, stdy = torch.std(datax), torch.std(datay)

# datax = (datax - meanx)/stdx
# datay = (datay - meany)/stdy

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

writer = SummaryWriter(
    "runs/Boston/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

w = torch.randn((x_train.size(1), y_train.size(1)), requires_grad=True)
b = torch.randn((y_train.size(1)), requires_grad=True)


for i in range(100):
    yhat = torch.mm(x_train, w) + b
    mse_train = torch.mean((yhat-y_train)**2)

    prediction = torch.mm(x_test, w) + b
    mse_test = toch.mean((prediction-y_test)**2)

    writer.add_scalar('lossBatch/train', mse_train, i)
    writer.add_scalar('lossBatch/test', mse_test, i)
    mse_train.backward()
    # print(w.grad, b.grad)

    with torch.no_grad():
        w -= eps*w.grad
        b -= eps*b.grad
    # w.data -= eps*w.grad
    # b.data -= eps*b.grad

    # reset les gradients
    w.grad.data.zero_()
    b.grad.data.zero_()


# gradient miniBatch

def gradient_mini_batch(X_train, Y_train, X_test, Y_test, epsilon, N, writter):
    w = torch.randn((X_train.size(1), Y_train.size(1)), requires_grad=True)
    b = torch.randn((Y_train.size(1)), requires_grad=True)

    optim = torch.optim.SGD(params=[w, b], lr=epsilon)
    optim.zero_grad()

    for i in range(100):
        draw = torch.randint(X_train.size(0), (1, N))
        yhat = torch.mm(X_train[draw], w) + b
        mse_train = torch.mean((yhat-Y_train[draw])**2)

        prediction = torch.mm(X_test, w) + b
        mse_test = torch.mean((prediction-Y_test)**2)

        writer.add_scalar('lossMiniBatch/train', mse_train, i)
        writer.add_scalar('lossMiniBatch/test', mse_test, i)
        mse_train.backward()

        optim.step()
        optim.zero_grad()


N = 40
gradient_mini_batch(x_train, y_train, x_test, y_test, eps, N, writer)


# gradient SGD

def gradient_SGD(X_train, Y_train, X_test, Y_test, epsilon, writter):
    w = torch.randn((X_train.size(1), Y_train.size(1)), requires_grad=True)
    b = torch.randn((Y_train.size(1)), requires_grad=True)

    optim = torch.optim.SGD(params=[w, b], lr=epsilon)
    optim.zero_grad()

    for i in range(100):
        draw = torch.randint(X_train.size(0), (1,))
        yhat = torch.mm(X_train[draw], w) + b
        mse_train = torch.mean((yhat-Y_train[draw])**2)

        prediction = torch.mm(X_test, w) + b
        mse_test = torch.mean((prediction-Y_test)**2)

        writer.add_scalar('lossSGD/train', mse_train, i)
        writer.add_scalar('lossSGD/test', mse_test, i)
        mse_train.backward()

        optim.step()
        optim.zero_grad()


gradient_SGD(x_train, y_train, x_test, y_test, eps, writer)
