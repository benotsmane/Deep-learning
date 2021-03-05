import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from math import inf
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
savepath = Path("../../../../../../../../tempory/BaseModel.pch")

class State:
    def __init__(self, model, optim ):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0

        
############## Affichage network data #########################

def hist_grad(writer, x, hid1, hid2, epoch):
    writer.add_histogram('g1', x.grad, epoch)
    writer.add_histogram('g2', hid1.grad, epoch)
    writer.add_histogram('g3', hid2.grad, epoch)


def hist_weights(writer, net, epoch):
    # fc1 and fc2 are Sequential
    writer.add_histogram('w1', net.fc1[0].weight, epoch)
    writer.add_histogram('w2', net.fc2[0].weight, epoch)
    writer.add_histogram('w3', net.fc3.weight, epoch)


def hist_sorties(writer, hid1, hid2, y_pred, epoch):
    writer.add_histogram('o1', hid1, epoch)
    writer.add_histogram('o2', hid2, epoch)
    writer.add_histogram('o3', y_pred, epoch)



###############################################################

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]


    
def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)



def collate(batch):
        """Collate function (pass to DataLoader's collate_fn arg).
        Args:
            batch (list): list of examples returned by __getitem__
        Returns:
            tuple: Three tensors: batch of padded documents, lengths of documents,
                and target classes.
        """
        data, target = list(zip(*batch))
        data = [torch.tensor(l) for l in data]
        lengths = torch.LongTensor([len(s) for s in data])
        # pad sequences
        return (pad_sequence(data, padding_value=0, batch_first=True), torch.LongTensor(target), lengths)

    

def compute_entropy(weigths):
    entropy = weigths.clone()
    entropy[entropy==0] = 1
    entropy = entropy * entropy.log()
    entropy = - entropy.sum(dim=1).mean()

    return entropy
    
    
    
#  TODO: 
class BaseModel(nn.Module):
    def __init__(self, glove_embedding, dim_hidden, dim_output):
        super(BaseModel, self).__init__()

        self.embedding = torch.nn.Embedding.from_pretrained(glove_embedding, freeze=True)
        self.layer1 = nn.Linear(glove_embedding.shape[1], dim_hidden)
        self.layer_out = nn.Linear(dim_hidden,dim_output)
        self.activ = nn.ReLU()
        

    def forward(self, X, lengths):
        mask = torch.arange(X.size(1), device=device)[None, : ] < lengths.view(-1)[ : ,None]
        X = self.embedding(X)
        X = X * mask.unsqueeze(2)
        X = torch.sum(X, 1) / lengths[:, None]
        output = self.activ(self.layer1(X.float()))
        output = self.layer_out(output)
        
        return output, torch.tensor(1/lengths)


    
class SimpleAttention(nn.Module):
    def __init__(self, glove_embedding, dim_hidden, dim_output):
        super(SimpleAttention, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(glove_embedding, freeze=True)

        self.classifieur = nn.Sequential(nn.Linear(glove_embedding.shape[1], dim_hidden),
                                         nn.ReLU(),
                                         nn.Linear(dim_hidden,dim_output))
                                         

        self.query = nn.Sequential(nn.Linear(glove_embedding.shape[1], dim_hidden),
                            nn.ReLU(),
                            nn.Linear(dim_hidden,glove_embedding.shape[1]))

        

    def forward(self, X, lengths):
        X_emb = self.embedding(X).float()
        alpha = (self.query(torch.ones(X_emb.size(2), device=device)) * X_emb).sum(2)
        
        mask = torch.arange(X.size(1), device=device)[None, : ] < lengths.view(-1)[ : ,None]
        alpha[~mask] = float('-inf')
        alpha = alpha.softmax(1)
        
        t_hat = alpha.unsqueeze(2) * X_emb
        t_hat = t_hat.sum(dim=1)
        output = self.classifieur(t_hat)
        
        return output, alpha
        

class ContextualAttention(nn.Module):
    def __init__(self, glove_embedding, dim_hidden, dim_output):
        super(ContextualAttention, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(glove_embedding, freeze=True)

        self.classifieur = nn.Sequential(nn.Linear(glove_embedding.shape[1], dim_hidden),
                                         nn.ReLU(),
                                         nn.Linear(dim_hidden,dim_output))
                                         

        self.value = nn.Sequential(nn.Linear(glove_embedding.shape[1], dim_hidden),
                            nn.ReLU(),
                            nn.Linear(dim_hidden,glove_embedding.shape[1]))
        
        self.query = nn.Sequential(nn.Linear(glove_embedding.shape[1], dim_hidden),
                            nn.ReLU(),
                            nn.Linear(dim_hidden,glove_embedding.shape[1]))

        

    def forward(self, X, lengths):
        X_emb = self.embedding(X).float()
        
        mask = torch.arange(X.size(1), device=device)[None, : ] < lengths.view(-1)[ : ,None]

        X_emb_mean = X_emb.clone()
        X_emb_mean[~mask] = torch.zeros(X_emb.size(2), device=device)
        X_emb_mean = X_emb_mean.sum(dim=1) / lengths[:, None]
        
        alpha = (self.query(X_emb_mean.unsqueeze(1)) * X_emb).sum(2)
    
        alpha[~mask] = float('-inf')
        alpha = alpha.softmax(1)
        t_hat = alpha.unsqueeze(2) * self.value(X_emb)
        t_hat = t_hat.sum(dim=1)
        output = self.classifieur(t_hat)
        
        return output, alpha




class ContextualAttentionRNN(nn.Module):
    def __init__(self, glove_embedding, dim_hidden, dim_output):
        super(ContextualAttentionRNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(glove_embedding, freeze=True)

        self.classifieur = nn.Sequential(nn.Linear(glove_embedding.shape[1], dim_hidden),
                                         nn.ReLU(),
                                         nn.Linear(dim_hidden,dim_output))
                                         

        self.value = nn.Sequential(nn.Linear(glove_embedding.shape[1], dim_hidden),
                            nn.ReLU(),
                            nn.Linear(dim_hidden,glove_embedding.shape[1]))
        
        self.query = nn.Sequential(nn.Linear(glove_embedding.shape[1], dim_hidden),
                            nn.ReLU(),
                            nn.Linear(dim_hidden,glove_embedding.shape[1]))

        self.lstm = nn.LSTM(glove_embedding.shape[1], glove_embedding.shape[1])

        

    def forward(self, X, lengths):
        X_emb = self.embedding(X).float()
        mask = torch.arange(X.size(1), device=device)[None, : ] < lengths.view(-1)[ : ,None]
        mask = mask.permute(1,0)
        H , _ = self.lstm(X_emb.permute(1,0,2))

        H_mean = H.clone()
        H_mean[~mask] = torch.zeros(H.size(2), device=device)
        H_mean = H_mean.sum(dim=0) / lengths[:, None]
        
        alpha = (self.query(H_mean.unsqueeze(0)) * H).sum(2)
    
        alpha[~mask] = float('-inf')
        alpha = alpha.softmax(0)
        t_hat = alpha.unsqueeze(2) * self.value(H)
        t_hat = t_hat.sum(dim=0)
        output = self.classifieur(t_hat)
        
        return output, alpha.permute(1,0)

class GradientDescent(object):

    def __init__(self, model, eps, epoch):
        self.model = model
        self.eps = eps
        self.epoch = epoch
        

    def descente_gradient(self, loader_train, loader_test, penality=0):

        #parametre à optimiser
        optim = torch.optim.Adam(self.model.parameters(), lr=self.eps)

        #scheduler
        #lr_sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

        criterion = nn.CrossEntropyLoss()
        #checkpoint
        
        if savepath.is_file():
            with savepath.open("rb") as fp:
                state = torch.load(fp)

        else:
            state = State(self.model, optim)



        rec_loss_train = [None]*self.epoch   #record train loss
        rec_loss_test = [None]*self.epoch
        rec_accuracy_train = [None]*self.epoch
        rec_accuracy_test = [None]*self.epoch

        for n_iter in range(state.epoch, self.epoch):
            state.model.train()
            cumul_loss = 0
            count_batch = 0
            accuracy=0
            
            for batch in loader_train:
                #Reinitialisation du gradient
                state.optim.zero_grad()
                
                outputs, alpha = state.model.forward(batch[0].to(device), batch[2].to(device))
                #entropy is tensor of size = batch_size
                
                
                target = batch[1].long().to(device)

                # Standard loss
                loss = criterion(outputs, target)

                # Loss with entropy regularisation : question 5
                if penality > 0:
                    entropy = compute_entropy(alpha)
                    loss += penality * entropy
                    
                loss.backward()
                
                #Mise à jour paramétres du modéle
                state.optim.step()
                state.iteration +=1

                #vidage GPU
                batch[0].cpu()
                target.cpu()
                outputs.cpu()

                cumul_loss += loss
                #Accuracy
                soft_m = torch.tensor(nn.functional.softmax(outputs, dim=1))
                label = torch.argmax(soft_m, dim=1)
                accuracy += (label==target.long()).sum().item() / len(target)

                count_batch +=1

            """
            with savepath.open("wb") as fp:
                state.epoch = state.epoch + 1
                torch.save(state, fp)
            """
            # on peut visualiser avec
            # tensorboard --logdir runs/
            #writer.add_scalar('Loss/train', cumul_loss, n_iter)
            #writer.add_scalar('accuracy/train', accuracy /count_batch, n_iter)
            # if... 
            #writer.add_histogram('entropy_train', entropy, epoch)
            
            # Sortie directe
            print(f"Itérations {n_iter}: loss/train {cumul_loss}")
            print(f"Itérations {n_iter}: ACCU/train {accuracy/count_batch}")
            rec_loss_train[n_iter]= cumul_loss
            rec_accuracy_train[n_iter] = accuracy /count_batch

            #Evalute loss in test
            with torch.no_grad():
                state.model.eval()
                cumul_loss = 0
                accuracy=0
                count_batch=0
                for batch in loader_test:
                    outputs, alpha = state.model.forward(batch[0].to(device), batch[2].to(device))
                    target = batch[1].long().to(device)
    
                    loss = criterion(outputs, target) #+ entropy
                    

                    #vidage GPU
                    batch[0].cpu()
                    outputs.cpu()
                    target.cpu()
                    
                    cumul_loss += loss

                    #Accuracy
                    soft_m = torch.tensor(nn.functional.softmax(outputs, dim=1))
                    label = torch.argmax(soft_m, dim=1)
                    accuracy += (label==target.long()).sum().item() / len(target)
                    count_batch +=1


            
            rec_loss_test[n_iter]= cumul_loss
            rec_accuracy_test[n_iter] = accuracy /count_batch
            # Sortie directe
            print(f"Itérations {n_iter}: loss/test {cumul_loss}")
            print(f"Itérations {n_iter}: ACCU/test {accuracy/count_batch}")
            #writer.add_scalar('Loss/test', cumul_loss, n_iter)
            #writer.add_scalar('accuracy/test', accuracy /count_batch, n_iter)

            #lr_sched.step()
        
        return (rec_loss_train, rec_loss_test), (rec_accuracy_train, rec_accuracy_test)



def plot(data, name_fig=""):
    #plt.yscale('log')
    plt.plot(data) 
    plt.xlabel('epoch')

    plt.ylabel('loss')
    plt.show()

    

if __name__=="__main__":
    #Parameters
    eps=1e-2
    epoch=10
    BATCH_SIZE=100
    dim_hidden_layer= 50
    dim_output=2

    word2id, glove_emb, train_dataset, test_dataset = get_imdb_data(embedding_size=50)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    
    #model = SimpleAttention(torch.tensor(glove_emb), dim_hidden_layer, dim_output)
    #model = BaseModel(torch.tensor(glove_emb), dim_hidden_layer, dim_output)
    model = ContextualAttention(torch.tensor(glove_emb), dim_hidden_layer, dim_output)
    #model = ContextualAttentionRNN(torch.tensor(glove_emb), dim_hidden_layer, dim_output)
    model = model.to(device)
    optimizer = GradientDescent(model, eps, epoch)
    
    loss, accuracy = optimizer.descente_gradient(train_loader, test_loader, penality=0.01)


    model.cpu()
    """
    plot(loss[0])
    plot(loss[1])
    plot(accuracy[0])
    plot(accuracy[1])
    
    word2id["PAD"] = word2id["__OOV__"] +1
    print(word2id["__OOV__"])
    print(glove_emb.shape)

    for batch in train_loader :
        print(batch[0].shape)
        print(batch[1].shape)
        break
    """
