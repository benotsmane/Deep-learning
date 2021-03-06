import string
import unicodedata
import torch
import torch.nn.functional as F
from utils import RNN, device, MonDataset, State
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt


LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

savepath = Path("model_generation.pch")


class RNN_generation(RNN):


    def __init__(self, dim_input, dim_output, dim_latent):
        super().__init__(dim_input, dim_output, dim_latent)


    def decode(self, h):
        D = torch.zeros(h.shape[0], h.shape[1], self.dim_output, device=device)
        for i in range(len(h)):
            D[i] =  self.w_d(h[i])
    
        return D

    def generation(self, X, seq_len):
        with torch.no_grad():
            h0 = torch.zeros(1, self.dim_latent)
            ht1 = self.forward(X.to(device), h0.to(device))
            yt1 = self.decode(torch.unsqueeze(ht1[-1],0))
            pred = F.softmax(yt1, dim=2).zero_()
            yt1 = F.softmax(yt1, dim=2).argmax(dim=2)
            pred[0].scatter_(1,yt1,1)
            for i in range(1, seq_len):
                ht1 = self.one_step(pred[0], ht1[-1])
                yt1 = self.decode(torch.unsqueeze(ht1,0))
                pred_t = F.softmax(yt1, dim=2).zero_()
                yt1 = F.softmax(yt1, dim=2).argmax(dim=2)
                pred_t[0].scatter_(1,yt1,1)
                pred = torch.cat((pred,pred_t))

            X.cpu()
            h0.cpu()
            ht1.cpu()
            yt1.cpu()
            pred.cpu()
             
        return pred.argmax(dim=2)
         
        
    


# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs_generation/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


def plot(data, name_fig=""):
    #plt.yscale('log')
    plt.plot(data) 
    plt.xlabel('epoch')

    plt.ylabel('loss')

 
    #plt.hist(tabNb)
    #plt.savefig(name_fig)
    plt.show()
    #plt.clear()



class GradientDescent(object):

    def __init__(self, model, eps, epoch, batch_size, dim_latent):
        self.model = model
        self.eps = eps
        self.epoch = epoch
        self.batch_size = batch_size
        self.dim_latent = dim_latent
        

    def descente_gradient(self, X_train, X_test):

        #parametre à optimiser
        optim = torch.optim.Adam(self.model.parameters(), lr=self.eps)

        #cross entropy loss
        loss_func = nn.CrossEntropyLoss(reduction='sum')

        #checkpoint
        if savepath.is_file():
            with savepath.open("rb") as fp:
                state = torch.load(fp)

        else:
            state = State(self.model, optim)
    

        rec_loss_train = [None]*epoch   #record train loss
        rec_loss_test = [None]*epoch
        for n_iter in range(state.epoch, self.epoch):
            cumul_loss = 0
            count_batch = 0
            for batch in X_train:
                #Reinitialisation du gradient
                state.optim.zero_grad()
                try:
                    batch = torch.stack(batch.split(100), dim=1)
                    h0 = torch.zeros(self.batch_size, self.dim_latent)

                except RuntimeError:
                    batch = torch.stack(batch.split(100)[:-1], dim=1)
                    h0 = torch.zeros(batch.shape[1], self.dim_latent)
                    
                H = state.model.forward(batch.to(device), h0.to(device))

                pred = state.model.decode(H)
                pred = pred.permute(0,2,1)
                real = batch[1:].argmax(dim=2)
                
                loss = loss_func(pred.narrow(0,0,len(pred)-1), real.to(device))
                loss.backward()
                
                #Mise à jour paramétres du modéle
                state.optim.step()
                state.iteration +=1

                cumul_loss +=loss 
                count_batch +=1

                #vidage GPU
                h0.cpu()
                H.cpu()
                batch.cpu()
                pred.cpu()
                real.cpu()
                

                
            with savepath.open("wb") as fp:
                state.epoch = state.epoch + 1
                torch.save(state, fp)

            # on peut visualiser avec
            # tensorboard --logdir runs/
            writer.add_scalar('Loss/train', cumul_loss, n_iter)

            # Sortie directe
            print(f"Itérations {n_iter}: loss {cumul_loss}")
            rec_loss_train[n_iter]= cumul_loss


            #Evalute loss in test
            with torch.no_grad():
                cumul_loss = 0
                count_batch = 0
                for batch in X_test:
                    try:
                        batch = torch.stack(batch.split(100), dim=1)
                        h0 = torch.zeros(self.batch_size, self.dim_latent)
                    except RuntimeError:
                        batch = torch.stack(batch.split(100)[:-1], dim=1)
                        h0 = torch.zeros(batch.shape[1], self.dim_latent)

                    H = state.model.forward(batch.narrow(0,0,len(batch)-1).to(device), h0.to(device))

                    pred = state.model.decode(torch.unsqueeze(H[-1],0))
                    real = batch.argmax(dim=2)
                    real = real[-1]
                    loss = loss_func(pred[0], real.to(device))

                    #vidage GPU
                    h0.cpu()
                    H.cpu()
                    batch.cpu()
                    pred.cpu()
                    real.cpu()
                    
                    cumul_loss +=loss 
                    count_batch +=1
                    
            writer.add_scalar('Loss/test', cumul_loss, n_iter)
            rec_loss_test[n_iter]= cumul_loss

        return rec_loss_train, rec_loss_test





if __name__=="__main__":    
    
    file_path = '../../data/trump_full_speech.txt'
    with open(file_path, "r") as fp:
        text = fp.read()

    dataset = string2code(text)
    #embedding one hot
    dataset_onehot = torch.FloatTensor(len(dataset), dataset.max()+1)
    dataset_onehot.zero_()
    dataset_onehot.scatter_(1,dataset.reshape(-1,1),1)

    #split data
    len_test = int(len(dataset)*0.3)
    i = torch.randint(len(dataset)-len_test, (1,))

    test = dataset_onehot[i:i+len_test]
    train = torch.cat((dataset_onehot[:i], dataset_onehot[i+len_test:]))

    data_train = DataLoader(MonDataset(train), batch_size=3200)
    data_test = DataLoader(MonDataset(test), batch_size=3200)

    #parameters
    batch_size=32
    eps=1e-3
    epoch=50
    dim_latent=20
    dim_input=dataset_onehot.shape[1]
    dim_output=dataset_onehot.shape[1]

    model = RNN_generation(dim_input, dim_output, dim_latent)
    model = model.to(device)
    optimizer = GradientDescent(model, eps, epoch, batch_size, dim_latent)
    
    cost_train, cost_test = optimizer.descente_gradient(data_train, data_test)

    
    plot(cost_train)
    plot(cost_test)


    #Generation
    with savepath.open("rb") as fp:
        state = torch.load(fp)


    len_seq = 10     # sequence fournit au model
    len_seq_predit = 10   # sequence à generer 
    ind = torch.randint(len(test)-len_seq-len_seq_predit,(1,))
    seq_test = test[ind:ind+len_seq]
    seq_test = torch.unsqueeze(seq_test,1)
    pred = state.model.generation(seq_test, len_seq_predit)
    print('Debut de chaine:\n', code2string(test[ind:ind+len_seq].argmax(dim=1).flatten()))
    print('Chaine à generer:\n', code2string(test[ind+len_seq:ind+len_seq+len_seq_predit].argmax(dim=1).flatten()))

    print('Chaine generé:\n', code2string(pred.flatten()))
