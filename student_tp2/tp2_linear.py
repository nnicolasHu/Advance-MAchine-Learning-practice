import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm


# TODO: 


## Regression lineaire

# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

eps = 0.05

writer = SummaryWriter("runs/Linear/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

w = torch.randn((x.size(1), y.size(1)), requires_grad=True)
b = torch.randn(y.size(1), requires_grad=True)

for i in range(100):
    yhat = x @ w + b
    mse = ((yhat-y)**2).sum()/x.size(0)

    writer.add_scalar('loss/train', mse, i)
    mse.backward()
    #print(w.grad, b.grad)

    with torch.no_grad():
        w -= eps*w.grad
        b -= eps*b.grad
    #w.data -= eps*w.grad
    #b.data -= eps*b.grad

    #reset les gradients
    w.grad.data.zero_()
    b.grad.data.zero_()
    

#yhat = datax.mm(w)+ b
#mse = torch.sum((yhat-datay)**2)/datax.size(0)
#mse.retain_grad()









