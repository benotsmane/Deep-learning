import csv
import numpy as np
import logging
import time
import string
from itertools import chain

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from textloader import *
from generate import *
import logging
import datetime
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt


#  TODO:  Implémenter maskedCrossEntropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
savepath = Path("../../../../../../../../tempory/LSTM.pch")
# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs_LSTM"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

class State:
    def __init__(self, model, optim ):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0
 

def maskedCrossEntropy(output, target, padcar):
    loss_func = nn.CrossEntropyLoss(reduction='none')
    loss = loss_func(output, target)
    loss *= padcar
    
    return torch.sum(loss)



class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self, dim_input, dim_output, dim_latent, vocab_size):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, dim_input)
        self.w_i = nn.Linear(dim_input, dim_latent)
        self.w_h = nn.Linear(dim_latent, dim_latent)
        self.w_d = nn.Linear(dim_latent, dim_output)
        self.activ = nn.Tanh()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_latent = dim_latent


    def one_step(self, X, h):
        X_emb = self.embedding(X.long())
        return self.activ(self.w_i(X_emb) + self.w_h(h))


    def forward(self, X, h):
        H = torch.zeros(X.shape[0], X.shape[1], h.shape[1], device=device)
        h_t_prec = h
        for i in range(len(X)):
            h_t = self.one_step(X[i], h_t_prec)
            h_t_prec = h_t
            H[i] = h_t

        return H


    def decode(self, h):
        D = torch.zeros(h.shape[0], h.shape[1], self.dim_output, device=device)
        for i in range(len(h)):
            D[i] =  self.w_d(h[i])
    
        return D


    


class LSTM(nn.Module):
    #  TODO:  Implémenter un LSTM
    def __init__(self, dim_input, dim_output, dim_latent, vocab_size):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, dim_input)
        self.w_f = nn.Linear(dim_latent+dim_input, dim_latent)
        self.w_i = nn.Linear(dim_latent+dim_input, dim_latent)
        self.w_c = nn.Linear(dim_latent+dim_input, dim_latent)
        self.w_o = nn.Linear(dim_latent+dim_input, dim_latent)
        self.w_d = nn.Linear(dim_latent, dim_output)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_latent = dim_latent
        self.activ_tanh = nn.Tanh()
        self.activ_sig = nn.Sigmoid()
        self.C = None
    

    def one_step(self, X, h):
        X_emb = self.embedding(X.long())
        mat_concat = torch.cat((h,X_emb), 1)
        ft = self.activ_sig(self.w_f(mat_concat))
        it = self.activ_sig(self.w_i(mat_concat))
        Ct = ft * self.C + it * self.activ_tanh(self.w_c(mat_concat))
        ot = self.activ_sig(self.w_o(mat_concat))
        ht = ot * self.activ_tanh(Ct)
        self.C = Ct
        return ht


    def forward(self, X, h):
        self.C = torch.zeros(X.shape[1], h.shape[1], device=device)
        H = torch.zeros(X.shape[0], X.shape[1], h.shape[1], device=device)
        h_t_prec = h
        for i in range(len(X)):
            h_t = self.one_step(X[i], h_t_prec)
            h_t_prec = h_t
            H[i] = h_t

        return H


    def decode(self, h):
        D = torch.zeros(h.shape[0], h.shape[1], self.dim_output, device=device)
        for i in range(len(h)):
            D[i] =  self.w_d(h[i])
    
        return D
    


class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    def __init__(self, dim_input, dim_output, dim_latent, vocab_size):
        super(GRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, dim_input)
        self.w_z = nn.Linear(dim_latent+dim_input, dim_latent, bias=False)
        self.w_r = nn.Linear(dim_latent+dim_input, dim_latent, bias=False)
        self.w = nn.Linear(dim_latent+dim_input, dim_latent, bias=False)
        self.w_d = nn.Linear(dim_latent, dim_output)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_latent = dim_latent
        self.activ_tanh = nn.Tanh()
        self.activ_sig = nn.Sigmoid()
    

    def one_step(self, X, h):
        X_emb = self.embedding(X.long())
        mat_concat = torch.cat((h,X_emb), 1)
        zt = self.activ_sig(self.w_z(mat_concat))
        rt = self.activ_sig(self.w_r(mat_concat))
        ht = (1 - zt) * h + zt * self.activ_tanh(self.w(torch.cat((rt * h, X_emb), 1)))
        return ht


    def forward(self, X, h):
        H = torch.zeros(X.shape[0], X.shape[1], h.shape[1], device=device)
        h_t_prec = h
        for i in range(len(X)):
            h_t = self.one_step(X[i], h_t_prec)
            h_t_prec = h_t
            H[i] = h_t

        return H


    def decode(self, h):
        D = torch.zeros(h.shape[0], h.shape[1], self.dim_output, device=device)
        for i in range(len(h)):
            D[i] =  self.w_d(h[i])
    
        return D


class GradientDescent(object):

    def __init__(self, model, eps, epoch, batch_size, dim_latent):
        self.model = model
        self.eps = eps
        self.epoch = epoch
        self.batch_size = batch_size
        self.dim_latent = dim_latent
        

    def descente_gradient(self, loader):

        #parametre à optimiser
        optim = torch.optim.Adam(self.model.parameters(), lr=self.eps)


        #checkpoint
        if savepath.is_file():
            with savepath.open("rb") as fp:
                state = torch.load(fp)

        else:
            state = State(self.model, optim)

        rec_loss_train = [None]*epoch   #record train loss
        rec_loss_test = [None]*epoch
        len_train = int(len(loader)*0.7)
        for n_iter in range(state.epoch, self.epoch):
            cumul_loss = 0
            count_batch = 0
            for i in range(len_train):
                #Reinitialisation du gradient
                state.optim.zero_grad()
                batch = next(iter(loader))
                h0 = torch.zeros(self.batch_size, self.dim_latent)
                    
                H = state.model.forward(batch.to(device), h0.to(device))

                pred = state.model.decode(H)
                pred = pred.permute(0,2,1)
                real = batch[1:].long()
                pad = real.clone()
                pad[pad>0] = 1
                
                loss = maskedCrossEntropy(pred.narrow(0,0,len(pred)-1), real.to(device), pad.to(device))
                loss.backward()
                
                #Mise à jour paramétres du modéle
                state.optim.step()
                state.iteration +=1

                #vidage GPU
                h0.cpu()
                H.cpu()
                batch.cpu()
                pred.cpu()
                real.cpu()
                pad.cpu()

                cumul_loss +=loss / torch.sum(pad)
                count_batch +=1


        
            with savepath.open("wb") as fp:
                state.epoch = state.epoch + 1
                torch.save(state, fp)
            
            # on peut visualiser avec
            # tensorboard --logdir runs/
            writer.add_scalar('Loss/train', cumul_loss, n_iter)
            #writer.add_histogram('grad couche output', state.model.w_d.weight.grad, n_iter)

            # Sortie directe
            print(f"Itérations {n_iter}: loss {cumul_loss}")
            rec_loss_train[n_iter]= cumul_loss


            #Evalute loss in test
            with torch.no_grad():
                cumul_loss = 0
                count_batch = 0
                for i in range(len_train, len(loader)):
                    batch = next(iter(loader))
                    h0 = torch.zeros(self.batch_size, self.dim_latent)

                    H = state.model.forward(batch.narrow(0,0,len(batch)-1).to(device), h0.to(device))

                    pred = state.model.decode(torch.unsqueeze(H[-1],0))
                    real = batch[-1].long()
                    pad = real.clone()
                    pad[pad>0] = 1
                    loss = maskedCrossEntropy(pred[0], real.to(device), pad.to(device))

                    #vidage GPU
                    h0.cpu()
                    H.cpu()
                    batch.cpu()
                    pred.cpu()
                    real.cpu()
                    pad.cpu()
                    
                    cumul_loss +=loss / torch.sum(pad)
                    count_batch +=1
                    
            writer.add_scalar('Loss/test', cumul_loss, n_iter)
            print(f"Itérations {n_iter}: loss/test {cumul_loss}")
            rec_loss_test[n_iter]= cumul_loss

        return rec_loss_train, rec_loss_test



def plot(data, name_fig=""):
    #plt.yscale('log')
    plt.plot(data) 
    plt.xlabel('epoch')

    plt.ylabel('loss')

 
    #plt.hist(tabNb)
    #plt.savefig(name_fig)
    plt.show()
    #plt.clear()



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot


if __name__=="__main__":    
    
    file_path = '../data/trump_full_speech.txt'
    with open(file_path, "r") as fp:
        text = fp.read()

    #parameters
    batch_size=512
    eps=1e-2
    epoch=100
    dim_latent=len(id2lettre)
    dim_input= len(id2lettre)
    dim_output= len(id2lettre)

    ds = TextDataset(text)
    loader = DataLoader(ds, collate_fn=collate_fn, shuffle=True, batch_size=batch_size)

    model = LSTM(dim_input, dim_output, dim_latent, len(id2lettre))
    model = model.to(device)
    optimizer = GradientDescent(model, eps, epoch, batch_size, dim_latent)
    
    cost_train, cost_test = optimizer.descente_gradient(loader)


    model.cpu()
    #plot(cost_train)
    #plot(cost_test)
    
    #Generation
    with savepath.open("rb") as fp:
        state = torch.load(fp)

    start_seq = "The world is "
    seq = generate_beam(state.model, None, None, EOS_IX, 5, start=start_seq, maxlen=50, nucleus=False)

    print('Sequence générée par beam search\n', seq)
    seq, _ = generate(state.model, None, None, EOS_IX, start=start_seq, maxlen=50)
    print('Sequence générée par sampling\n', seq)
