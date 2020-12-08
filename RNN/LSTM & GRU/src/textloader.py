import sys
from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
import re
from typing import List
import torch
import random

PAD_IX = 0
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '' ##NULL CHARACTER
id2lettre[EOS_IX] = '|'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def normalize(s):
    """ enlève les accents et les majuscules """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

#  TODO: 

class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        self.sentences = list(map(str.strip, text.split(".")))

    def __len__(self):
        #  TODO:  Nombre de phrases
        return len(self.sentences)

    def __getitem__(self, i):
        #  TODO:
        return string2code(self.sentences[i])
        

def collate_fn(samples: List[List[int]]):
    #  TODO:  Renvoie un batch
    max_len_sentence = len(max(samples, key=len))+2
    batch = torch.zeros(len(samples), max_len_sentence)
    for i, sentence in enumerate(samples):
        samples[i] = torch.cat((sentence, string2code('.|')))
        if len(samples[i]) < max_len_sentence:
            padding = torch.zeros(max_len_sentence - len(samples[i])) 
            samples[i] = torch.cat((samples[i], padding))
        batch[i] = samples[i]

    return batch.t()
    
    
    

    

if __name__ == "__main__":
    test = "C'est. Un. Test."
    ds = TextDataset(test)
    loader = DataLoader(ds, collate_fn=collate_fn, batch_size=3)
    data = next(iter(loader))

    # Longueur maximum
    assert data.shape == (7, 3)

    # e dans les deux cas
    assert data[2, 0] == data[1, 2]
    # les chaînes sont identiques
    assert test == " ".join([code2string(s).replace("|","") for s in data.t()])

