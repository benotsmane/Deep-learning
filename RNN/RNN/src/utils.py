import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import torch.nn.functional as F
from torch.utils.data import Dataset
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


def fill_na(mat):
    ix,iy = np.where(np.isnan(mat))
    for i,j in zip(ix,iy):
        if np.isnan(mat[i+1,j]):
            mat[i,j]=mat[i-1,j]
        else:
            mat[i,j]=(mat[i-1,j]+mat[i+1,j])/2.
    return mat
"""
def read_temps(path):
    return torch.tensor(fill_na(np.array(pd.read_csv(path).iloc[:11116,1:])),dtype=torch.float)
"""

def read_temps(path):
    """Lit le fichier de températures"""
    data = []
    with open(path, "rt") as fp:
        reader = csv.reader(fp, delimiter=',')
        next(reader)
        for row in reader:
            data.append([float(x) if x != "" else float('nan') for x in row[1:]])
    return torch.tensor(fill_na(np.array(data)), dtype=torch.float)



class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1

    def __init__(self, dim_input, dim_output, dim_latent):
        super(RNN, self).__init__()

        self.w_i = nn.Linear(dim_input, dim_latent)
        self.w_h = nn.Linear(dim_latent, dim_latent)
        self.w_d = nn.Linear(dim_latent, dim_output)
        self.activ = nn.Tanh()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_latent = dim_latent


    def one_step(self, X, h):
        return self.activ(self.w_i(X) + self.w_h(h))

    def forward(self, X, h):
        H = torch.zeros(X.shape[0], X.shape[1], h.shape[1], device=device)
        h_t_prec = h
        for i in range(len(X)):
            h_t = self.one_step(X[i], h_t_prec)
            h_t_prec = h_t
            H[i] = h_t

        return H
    
        
#  TODO:  Implémenter les classes Dataset ici


class State:
    def __init__(self, model, optim ):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0
        



class MonDataset(Dataset):
    def __init__(self, X):
        self.X = X
        


    def __getitem__(self, index):
        return self.X[index]


    def __len__(self):
        return len(self.X)



class ConstructBatch(object):

    def __init__(self, X, Y, batch_size, len_max_seq):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.len_max_seq = len_max_seq



    def batches(self, seq_length, nb_features, nb_classe):
        batch = torch.zeros(self.batch_size, seq_length, nb_features)
        labels = torch.zeros(self.batch_size)

        ind_classe=torch.randint(nb_classe,(self.batch_size,))
        start_sequence = torch.randint(len(self.X)-seq_length, (self.batch_size,))
        for n in range(self.batch_size):
            batch[n] = self.X[start_sequence[n]:start_sequence[n]+seq_length, ind_classe[n]]

            labels[n] = ind_classe[n]
            

        return torch.transpose(batch, 1, 0), labels


    
    def construct_data(self, nb_batch, nb_features, nb_classe, seq=-1):
        if seq >= 0:
            seq_len = torch.tensor(seq).expand(nb_batch)
            
        else :
            seq_len = torch.randint(10, self.len_max_seq, (nb_batch,))
            
        X_data, Y_data = self.batches(seq_len[0], nb_features, nb_classe) 
        for n in range(1, nb_batch):
            X_batch, Y_batch = self.batches(seq_len[n], nb_features, nb_classe)
            X_data = torch.cat([X_data, X_batch], dim=0)
            Y_data = torch.cat([Y_data, Y_batch], dim=0)

        return X_data, Y_data, seq_len

    
        
        
class DataFile(object):
    def __init__(self, X, Y, seq_len):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
    
