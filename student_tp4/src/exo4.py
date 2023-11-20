import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN, device
import datetime
from tqdm import tqdm

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO: 

def embedding(X, lenD):
    #lenD: length of the dictionary
    #lenF: chosen final dimension
    #X: batch x length

    #One Hot Encoding
    onehot_matrix = torch.zeros(X.shape[0], X.shape[1], lenD)
    for b in range(X.shape[0]):
        for l in range(X.shape[1]):
            onehot_matrix[b, l, X[b, l]] = 1
    return onehot_matrix

writer = SummaryWriter("runs/Trump/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


latent = 20
BATCH_SIZE = 32
LENGTH = 100
PATH = "../data/"

data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size= BATCH_SIZE, shuffle=True)

lenD = len(lettre2id)
output = lenD
lenF = 50 #arbitrary choice: length of n' in the Embedding phase

n_epochs = 500
#emb = lambda X: embedding(X, lenD, lenF)
emb_linear = nn.Linear(lenD, lenF)
model = RNN(lenF, latent, output)
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in tqdm(range(n_epochs)):
    compteur = 0
    for (X, y) in data_trump:
        #print(X.shape, y.shape)
        X_one_hot = embedding(X, lenD)
        X_emb = emb_linear(X_one_hot)
        H = model.forward(X_emb)
        yhat = model.decode(H)
        yhat = torch.permute(yhat, (1,2,0))
        loss = loss_fun(yhat, y)
        with torch.no_grad():
            _, idx = torch.max(yhat, 1)
            accuracy = torch.sum(idx == y) / y.numel()
        
        writer.add_scalar('lossRNN/train', loss, epoch*len(data_trump) + compteur)
        writer.add_scalar('accuracy/train', accuracy, epoch*len(data_trump) + compteur)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        compteur +=1


len_generate = 20
start = "Chi"
h = torch.zeros(1, latent)
for char in start:
    one_hot = embedding(string2code(char).unsqueeze(0), lenD)
    X_emb = emb_linear(one_hot.squeeze(0))
    h = model.one_step(X_emb, h)


while (len(start) < len_generate):
    yhat = model.decode(h)
    _, idx = torch.max(nn.Softmax(dim=1)(yhat), 1)
    new = id2lettre[idx.item()]
    #new='a'
    start += new

    one_hot = embedding(string2code(new).unsqueeze(0), lenD)
    X_emb = emb_linear(one_hot.squeeze(0))
    h = model.one_step(X_emb, h)
    print(start)