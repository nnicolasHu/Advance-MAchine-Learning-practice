
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO: 

# Question 2
def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    mask = (target != padcar)
    output = output.permute(1, 2, 0)
    loss_fn = CrossEntropyLoss(reduction='none')
    loss = loss_fn(output, target)
    masked_loss = loss * mask
    final_loss = masked_loss.sum() / mask.sum()
    return final_loss

class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self, dimx, dim_latent, dim_output):
        super().__init__()
        self.dimx = dimx
        self.dim_latent = dim_latent
        self.dim_output = dim_output

        self.l1 = nn.Linear(dimx, dim_latent, bias=False)
        self.l2 = nn.Linear(dim_latent, dim_latent)
        self.activation1 = nn.Tanh()

        # Le decodeur est independant de la classe et peut etre implemente en dehors
        self.l_decode = nn.Linear(dim_latent, dim_output)


    def one_step(self, x, h):
        """
        input : - x_t de dimention (batch x dimx)
                - h de dimension (batch x latent)
        output :- matrice de tous les etats caches (batch x latent)
        """
        return self.activation1(self.l1(x) + self.l2(h))
    

    def forward(self, x, h_0=None):
        """
        input : - x de dimention (batch x length x dimx)
                - h_0 de dimension (batch x latent)
        output :- matrice de tous les etats caches (lenght x batch x latent)
        """
        if h_0 is None:
            h_0 = torch.zeros(x.shape[0], self.dim_latent)
        res = [h_0]
        

        for i in range(1, x.size(1)):
            res.append(self.one_step(x[:,i-1,:], res[i-1]))
        
        return torch.stack(res, dim=0)


    def decode(self, h):
        """
        input : - h de dimention (batch x latent)
        output :- matrice des vecteurs one-hot sortie de taille (batch x dim_output)
        """
        return self.l_decode(h)



class LSTM(RNN):
    #  TODO:  Implémenter un LSTM


class GRU(nn.Module):
    #  TODO:  Implémenter un GRU



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
