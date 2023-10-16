# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/baskiotis/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """

    def __init__(self):
        self._saved_tensors = ()

    def save_for_backward(self, *args):
        self._saved_tensors = args

    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""

    @staticmethod
    def forward(ctx, yhat, y):
        # Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        #  TODO:  Renvoyer la valeur de la fonction
        # MSE = 1/n * ||ŷ - y||**2
        return torch.mean((yhat - y) ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        # Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        size = y.size(0)
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        dyhat = 2 / size * grad_output * (yhat - y)
        # dy = -2/size*grad_output*(yhat-y)
        return (dyhat, -dyhat)


#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE
class Linear(Function):
    """Début d'implementation de la fonction MSE"""

    @staticmethod
    def forward(ctx, W, X, b):
        # Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(W, X, b)

        #  TODO:  Renvoyer la valeur de la fonction
        return torch.mm(X, W) + b

    @staticmethod
    def backward(ctx, grad_output):
        # Calcul du gradient du module par rapport a chaque groupe d'entrées
        W, X, b = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        dW = torch.mm(X.T, grad_output)
        dX = torch.mm(grad_output, W.T)
        db = torch.sum(grad_output, 0)
        return (dW, dX, db)


# Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply
