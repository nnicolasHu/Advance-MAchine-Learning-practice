import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05

writer = SummaryWriter(
    "runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
for n_iter in range(100):
    # TODO:  Calcul du forward (loss)
    ctx_lin = Context()
    yhat = Linear.forward(ctx_lin, w, x, b)

    ctx_mse = Context()
    loss = MSE.forward(ctx_mse, yhat, y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    # TODO:  Calcul du backward (grad_w, grad_b)
    grad_output = torch.ones(1)  # gradient de Id?
    dyhat, _ = MSE.backward(ctx_mse, grad_output)
    dw, _, db = Linear.backward(ctx_lin, dyhat)

    # TODO:  Mise à jour des paramètres du modèle
    w -= epsilon*dw
    b -= epsilon*db

# visusalisation
# tensorboard --logdir runs/
# depuis student_tp1
