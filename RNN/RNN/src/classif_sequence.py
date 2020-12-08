from utils import read_temps, RNN, device, MonDataset, ConstructBatch, DataFile, State
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

#saving model
savepath = Path("model_classif.pch")

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs_classif/train_fixe_test_fixe/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



class RNN_classif(RNN):


    def __init__(self, dim_input, dim_output, dim_latent):
        super().__init__(dim_input, dim_output, dim_latent)


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
        

    def descente_gradient(self, X_train, Y_train, seq_len_train, X_test, Y_test, seq_len_test):
        """
        X_train : dim(len_sequence, batch_size, nb_feature) contains all batchs concat following dim 0

        Y_train : 1D tensor,  
                  32 first label for the first batch, 
                  from 32 to 64 for second batch etc...
        
        seq_len_train: 1D tensor, 
                       seq_len_train[0]=25 ----> 1st batch sequence length = 25
                       seq_len_train[1]=5 ----> 2nd batch sequence length = 5 
        """

        #parametre à optimiser
        optim = torch.optim.SGD(self.model.parameters(), lr=self.eps)

        #cross entropy loss
        loss_func = nn.CrossEntropyLoss()

        #checkpoint
        if savepath.is_file():
            with savepath.open("rb") as fp:
                state = torch.load(fp)

        else:
            state = State(self.model, optim)
    

        rec_loss_train = [None]*epoch   #record train loss
        rec_loss_test = [None]*epoch
        rec_cost = [None]*epoch
        for n_iter in range(state.epoch, self.epoch):
            cumul_loss = 0
            count_batch = 0
            ind = 0
            for i, len_seq in enumerate(seq_len_train):
                #Reinitialisation du gradient
                state.optim.zero_grad()
                
                X_batch = X_train[ind:ind+len_seq]
                ind += len_seq
                h0 = torch.zeros(self.batch_size, self.dim_latent)
                H = state.model.forward(X_batch.to(device), h0.to(device))

                Y_hat = state.model.decode(torch.unsqueeze(H[-1],0))
                Y_batch = Y_train[i*self.batch_size:(i+1)*self.batch_size]
                
                loss = loss_func(Y_hat[0], Y_batch.long().to(device))
                loss.backward()
                
                #Mise à jour paramétres du modéle
                state.optim.step()
                state.iteration +=1

                cumul_loss +=loss
                count_batch +=1

                #vidage GPU
                h0.cpu()
                H.cpu()
                X_batch.cpu()
                Y_hat.cpu()
                Y_batch.cpu()
                

            with savepath.open("wb") as fp:
                state.epoch = state.epoch + 1
                torch.save(state, fp)

            # on peut visualiser avec
            # tensorboard --logdir runs/
            writer.add_scalar('Loss/train', cumul_loss/count_batch, n_iter)

            # Sortie directe
            print(f"Itérations {n_iter}: loss {cumul_loss/count_batch}")
            rec_loss_train[n_iter]= cumul_loss/count_batch


            #Evalute loss in test
            with torch.no_grad():
                cost_0_1 = 0
                cumul_loss = 0
                count_batch = 0
                ind = 0
                for i, len_seq in enumerate(seq_len_test):
                    X_batch = X_test[ind:ind+len_seq]
                    ind += len_seq
                    h0 = torch.zeros(self.batch_size, self.dim_latent)
                    H = state.model.forward(X_batch.to(device), h0.to(device))
                    Y_hat_test = state.model.decode(torch.unsqueeze(H[-1],0))
                    Y_batch = Y_test[i*self.batch_size:(i+1)*self.batch_size]
                    loss_test = loss_func(Y_hat_test[0], Y_batch.long().to(device))

                    #Cost 0-1
                    Y_hat_test = Y_hat_test.cpu()
                    Y_batch = Y_batch.cpu()
                    soft_m = torch.tensor(nn.functional.softmax(Y_hat_test[0]))
                    label = torch.argmax(soft_m, dim=1)
                    cost_0_1 += (label!=Y_batch.long()).sum().item() / self.batch_size

                    cumul_loss +=loss_test
                    count_batch +=1

                    #vidage GPU
                    h0.cpu()
                    H.cpu()
                    X_batch.cpu()
                    
            writer.add_scalar('Loss/test', cumul_loss/count_batch, n_iter)
            rec_loss_test[n_iter]= cumul_loss/count_batch
            writer.add_scalar('Pourcentage erreur classif', cost_0_1 / count_batch, n_iter)
            rec_cost[n_iter]= cost_0_1 / count_batch


        return rec_loss_train, rec_loss_test, rec_cost
        


def plot(data, name_fig=""):
    #plt.yscale('log')
    plt.plot(data) 
    plt.xlabel('epoch')

    plt.ylabel('loss')

 
    #plt.hist(tabNb)
    #plt.savefig(name_fig)
    plt.show()
    #plt.clear()

    

if __name__ == "__main__":
    train_savepath = Path("train.pch")
    test_savepath = Path("test.pch")

    batch_size = 32
    max_len_sequence = 200
    len_test_sequence = 100
    dim_input=1
    dim_output=10
    dim_latent = 20
    epoch = 30
    eps = 1e-3
    
    if train_savepath.is_file():
        with train_savepath.open("rb") as fp:
            train = torch.load(fp)
            X_train = train.X
            Y_train = train.Y
            seq_len_train = train.seq_len


    if test_savepath.is_file():
        with test_savepath.open("rb") as fp:
            test = torch.load(fp)
            X_test = test.X
            Y_test = test.Y
            seq_len_test = test.seq_len
            
    else:
        data_train = read_temps('../../data/tempAMAL_train.csv')
        data_test = read_temps('../../data/tempAMAL_test.csv')

        train = torch.unsqueeze(data_train.narrow(1,0,10), 2)
        test = torch.unsqueeze(data_test.narrow(1,0,10), 2)

        #Normalise data
        Max = torch.max(train) 
        Min = torch.min(train)
        train = (train - Min)/(Max-Min)
        test = (test - Min)/(Max-Min)
        

        batch_construct = ConstructBatch(train, None, batch_size, max_len_sequence)
        X_train, Y_train, seq_len_train = batch_construct.construct_data(len(train)//batch_size, 1, 10, seq=len_test_sequence)

        batch_construct = ConstructBatch(test, None, batch_size, max_len_sequence)
        X_test, Y_test, seq_len_test = batch_construct.construct_data(len(test)//batch_size, 1, 10)
        
        
        file_train = DataFile(X_train, Y_train, seq_len_train)
        file_test = DataFile(X_test, Y_test, seq_len_test)
    
        with train_savepath.open("wb") as fp:
            torch.save(file_train, fp)
    
        with test_savepath.open("wb") as fp:
            torch.save(file_test, fp)


    model = RNN_classif(dim_input, dim_output, dim_latent)
    model = model.to(device)
    optimizer = GradientDescent(model, eps, epoch, batch_size, dim_latent)
    
    cost_train, cost_test, cost = optimizer.descente_gradient(X_train, Y_train, seq_len_train, X_test, Y_test, seq_len_test)
    
    plot(cost_train)
    plot(cost_test)
    plot(cost)
