# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        #ctx.save_for_backward(yhat, y)
        
        return (1/yhat.shape[0]) * torch.pow(torch.norm(yhat-y), 2)

        #  TODO:  Renvoyer la valeur de la fonction

   

mse = MSE.apply

#  TODO:  Implémenter la fonction Linear(X, W, b)

class Linear(Function):
    """Début d'implementation de la fonction Linear"""
    @staticmethod
    def forward(x, w, b):
        ## Garde les valeurs nécessaires pour le backwards
        #ctx.save_for_backward(x, w, b)
        return torch.mm(x,w) + b
        #  TODO:  Renvoyer la valeur de la fonction

    

#mse = MSE.apply

linear = Linear.apply
