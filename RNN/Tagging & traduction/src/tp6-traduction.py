import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import time
import re
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(level=logging.INFO)


FILE = "../../data/en-fra.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

savepath = Path("../../../../../../../../tempory/GRU_traduction1.pch")

writer = SummaryWriter("runs_trad/runsV2"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
            
    def __len__(self):
        return len(self.sentences)

    
    def __getitem__(self,i):
        return self.sentences[i]



def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    
    return pad_sequence(orig) ,pad_sequence(dest)   #, o_len, d_len



class State:
    def __init__(self, model, optim ):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0



class GRU(nn.Module):
    #  TODO:  Implémenter un LSTM
    def __init__(self, dim_input, dim_output, dim_latent, orig_vocab_size, dest_vocab_size):
        super(GRU, self).__init__()
        #Encodage
        self.embedding_orig = nn.Embedding(orig_vocab_size, dim_input)
        self.gru_encode = nn.GRU(dim_input, dim_latent)
        #decodage
        self.embedding_dest = nn.Embedding(dest_vocab_size, dim_input)
        self.activ = nn.ReLU()
        self.gru_decode = nn.GRU(dim_input, dim_latent)
        self.hidden_2_out = nn.Linear(dim_latent, dim_output)
        #param Dim
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_latent = dim_latent
    


    def encode(self, X):
        X_emb = self.embedding_orig(X.long())
        _ , h_n = self.gru_encode(X_emb)
        
        return h_n


    def decode(self, X, h, SOS=2, EOS=1, contraint=True): 
        if contraint:
            X_emb = self.embedding_dest(X.long())
            X_emb = self.activ(X_emb)
            X_0 = self.activ(self.embedding_dest(torch.tensor(SOS).expand(1,X.shape[1]).to(device)))
            _, H_0 = self.gru_decode(X_0, h)
            
            H, _ = self.gru_decode(X_emb[:len(X_emb)-1], H_0)
            H = torch.cat((H_0, H))
            D = self.hidden_2_out(H)
            outputs = F.softmax(D, dim=2).argmax(dim=2)
            X_0.cpu()

        else:
            outputs, D = self.generate(h, lenseq=len(X))

        
        return outputs, D


    
    def generate(self, h, lenseq=None, SOS=2, EOS=1, MAX_LEN=20):
        X_0 = self.activ(self.embedding_dest(torch.tensor(SOS).expand(1,h.shape[1]).to(device)))
        _, H = self.gru_decode(X_0, h)
        
        D = self.hidden_2_out(H)
        outputs = F.softmax(D, dim=2).argmax(dim=2)
        if lenseq is not None:
            for i in range(lenseq-1):
                X_t = self.activ(self.embedding_dest(outputs[-1].unsqueeze(0)))
                _, h_t = self.gru_decode(X_t, H[-1].unsqueeze(0))
                H = torch.cat((H, h_t))
                D = torch.cat((D, self.hidden_2_out(H[-1].unsqueeze(0))))
                outputs_t = F.softmax(D[-1].unsqueeze(0), dim=2).argmax(dim=2)
                outputs_t[0][outputs[-1]==0]=0
                outputs_t[0][outputs[-1]==1]=0
                outputs = torch.cat((outputs, outputs_t))

        
        else :
            i=0
            while outputs[-1]!=EOS and i<MAX_LEN:
                X_t = self.activ(self.embedding_dest(outputs[-1].unsqueeze(0)))
                _, h_t = self.gru_decode(X_t, H[-1].unsqueeze(0))
                H = torch.cat((H, h_t))
                D = torch.cat((D, self.hidden_2_out(H[-1].unsqueeze(0))))
                outputs_t = F.softmax(D[-1].unsqueeze(0), dim=2).argmax(dim=2)
                outputs = torch.cat((outputs, outputs_t))
                i+=1

        X_0.cpu()
        
        return outputs, D
        


def get_token_accuracy(targets, outputs, ignore_index=None):
    """ Get the accuracy token accuracy between two tensors.
    """

    n_correct = 0.0
    n_total = 0.0
    for target, output in zip(targets, outputs):
        prediction = output
        
        if ignore_index is not None:
            mask = target.ne(ignore_index)
            n_correct += prediction.eq(target).masked_select(mask).sum().item()
            n_total += mask.sum().item()
        else:
            n_total += len(target)
            n_correct += prediction.eq(target).sum().item()

    return n_correct / n_total, n_correct, n_total



class GradientDescent(object):

    def __init__(self, model, eps, epoch):
        self.model = model
        self.eps = eps
        self.epoch = epoch
        

    def descente_gradient(self, loader_train, loader_test, pad_idx=0, SOS=2, EOS=1):

        #parametre à optimiser
        optim = torch.optim.Adam(self.model.parameters(), lr=self.eps)

        #scheduler
        lr_sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')
        #checkpoint
        if savepath.is_file():
            with savepath.open("rb") as fp:
                state = torch.load(fp)

        else:
            state = State(self.model, optim)

        rec_loss_train = [None]*epoch   #record train loss
        rec_loss_test = [None]*epoch
        rec_accuracy_train = [None]*epoch
        rec_accuracy_test = [None]*epoch
        proba = torch.tensor(0.8)
        for n_iter in range(state.epoch, self.epoch):
            cumul_loss = 0
            count_batch = 0
            accuracy=0
            """
            if n_iter%5==0 and proba >0.3:
                proba -= 0.5
                
            n = torch.bernoulli(proba)
            if n==1:
                contraint=True
            else:
                contraint=False
            """
            for batch in loader_train:
                #Reinitialisation du gradient
                state.optim.zero_grad()
                
                H = state.model.encode(batch[0].to(device))
                target = batch[1].to(device)
                #outputs, D = state.model.decode(target, H, SOS=SOS, EOS=EOS, contraint=contraint)

                p = torch.randn(1)
                if p < 0.5:
                    outputs, D = state.model.decode(target, H, SOS=SOS, EOS=EOS, contraint=True)
                else:
                    outputs, D = state.model.decode(target, H, SOS=SOS, EOS=EOS, contraint=False)
    
                loss = criterion(D.permute(0,2,1), target)
                loss.backward()
                
                #Mise à jour paramétres du modéle
                state.optim.step()
                state.iteration +=1

                #vidage GPU
                H.cpu()
                batch[0].cpu()
                target.cpu()
                D.cpu()
                outputs.cpu()

                cumul_loss +=loss / len(target[0])
                #Accuracy
                accu_tmp, _, _ = get_token_accuracy(target, outputs, ignore_index=pad_idx)
                
                accuracy += accu_tmp
                count_batch +=1

        
            with savepath.open("wb") as fp:
                state.epoch = state.epoch + 1
                torch.save(state, fp)
            
            # on peut visualiser avec
            # tensorboard --logdir runs/
            writer.add_scalar('Loss/train', cumul_loss, n_iter)
            writer.add_scalar('accuracy/train', accuracy /count_batch, n_iter)

            # Sortie directe
            print(f"Itérations {n_iter}: loss/train {cumul_loss}")
            rec_loss_train[n_iter]= cumul_loss
            rec_accuracy_train[n_iter] = accuracy /count_batch

            #Evalute loss in test
            with torch.no_grad():
                cumul_loss = 0
                accuracy=0
                count_batch=0
                for batch in loader_test:
                    H = state.model.encode(batch[0].to(device))
                    
                    target = batch[1].to(device)
                    outputs, D = state.model.decode(target, H, SOS=SOS, EOS=EOS, contraint=False)

                    loss = criterion(D.permute(0,2,1), target)

                    

                    #vidage GPU
                    H.cpu()
                    batch[0].cpu()
                    D.cpu()
                    outputs.cpu()
                    target.cpu()
                    
                    cumul_loss +=loss / len(target[0])

                    #Accuracy
                    accu_tmp, _, _ = get_token_accuracy(target, outputs, ignore_index=pad_idx)
                    accuracy += accu_tmp
                    count_batch +=1


            
            rec_loss_test[n_iter]= cumul_loss
            rec_accuracy_test[n_iter] = accuracy /count_batch
            # Sortie directe
            print(f"Itérations {n_iter}: loss/test {cumul_loss}")
            print(f"Itérations {n_iter}: ACCU/test {accuracy/count_batch}")
            writer.add_scalar('Loss/test', cumul_loss, n_iter)
            writer.add_scalar('accuracy/test', accuracy /count_batch, n_iter)

            lr_sched.step()
        
        return (rec_loss_train, rec_loss_test), (rec_accuracy_train, rec_accuracy_test)
            

def plot(data, name_fig=""):
    #plt.yscale('log')
    plt.plot(data) 
    plt.xlabel('epoch')

    plt.ylabel('loss')

 
    #plt.hist(tabNb)
    #plt.savefig(name_fig)
    plt.show()
    #plt.clear()


if __name__=="__main__":

    MAX_LEN=100
    BATCH_SIZE=128

    with open(FILE) as f:
        lines = f.readlines()

    lines = [lines[x] for x in torch.randperm(len(lines))]
    idxTrain = int(0.8*len(lines))
    
    vocEng = Vocabulary(True)
    vocFra = Vocabulary(True)
    
    datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
    datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

    train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datatest, collate_fn=collate, batch_size=BATCH_SIZE)

    eps=1e-3
    epoch=50
    dim_latent=1000
    dim_input= 1000
    dim_output= len(vocFra)
    orig_vocab_size= len(vocEng)
    dest_vocab_size= len(vocFra)
    """
    model = GRU(dim_input, dim_output, dim_latent, orig_vocab_size, dest_vocab_size)
    
    model = model.to(device)
    optimizer = GradientDescent(model, eps, epoch)
    
    loss, accuracy = optimizer.descente_gradient(train_loader, test_loader, pad_idx=0, SOS=2, EOS=1)

    
    model.cpu()

    plot(loss[0])
    plot(loss[1])
    plot(accuracy[0])
    plot(accuracy[1])
    
    """
    #Generation
    with savepath.open("rb") as fp:
        state = torch.load(fp)
    with torch.no_grad():    

        for batch in test_loader:
            X = batch[0][:,0]
            X = torch.tensor(X).unsqueeze(1)
            H = state.model.encode(X.to(device))
            output, _ = state.model.generate(H)
            seq_in = vocEng.getwords(X.flatten())
            seq_out = vocFra.getwords(output.flatten())
            print(" ".join(seq_in))
            print(" ".join(seq_out))
            break
