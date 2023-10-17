import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
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
    

class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

