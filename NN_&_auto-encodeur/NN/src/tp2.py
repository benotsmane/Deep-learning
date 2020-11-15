import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from descente import MSE, Linear


## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
#import datamaestro
from tqdm import tqdm


writer = SummaryWriter("runs/Descente_mini-batch_1")

#data=datamaestro.prepare_dataset("edu.uci.boston")
#colnames, datax, datay = data.data()

#Getting boston data
datax, datay = load_boston(return_X_y=True)
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

#Init param
w = torch.randn(datax.shape[1], datay.shape[1], requires_grad=True)
b = torch.randn(datay.shape[1], requires_grad=True)
epsilon = 0.0000001
epoch = 100

#split data
idx = torch.randperm(len(datax))
x_train, x_test = torch.split(datax, int(len(idx)*0.8))
y_train, y_test = torch.split(datay, int(len(idx)*0.8))


def normalize_data(train, test):
    mean = torch.mean(train, 0, True)
    std = torch.std(train, 0, True)
    norm_train = (train - mean) / std
    norm_test = (test - mean) / std

    return norm_train, norm_test
    



def descente_grad_batch(train, test, epsilon=0.01, epoch=1000, norm_data=True):
    rec_loss_train = [None]*epoch
    rec_loss_test = [None]*epoch
    #normalisation des données 
    if(norm_data):
        train, test = normalize_data(train, test)
    
    for n_iter in range(epoch):
    
        y_hat = Linear.forward(train, w, b)

        loss = MSE.forward(y_hat, y_train)

        # `loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', loss/len(train), n_iter)
        rec_loss_train[n_iter]=loss/len(train)

        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss/len(train)}")

        ##  TODO:  Calcul du backward (grad_w, grad_b)
        loss.backward()
        
        ##  TODO:  Mise à jour des paramètres du modèle
        w.data = w.data - epsilon * w.grad.data
        b.data = b.data - epsilon * b.grad.data
        
        w.grad.data.zero_()
        b.grad.data.zero_()

        with torch.no_grad():
            y_hat_test = Linear.forward(test, w, b)
            loss_test = MSE.forward(y_hat_test, y_test)
            writer.add_scalar('Loss/test', loss_test/len(test), n_iter)
            rec_loss_test[n_iter]=loss_test/len(test)

    return rec_loss_train, rec_loss_test


def descente_grad_stochastic(train, test, epsilon=0.01, epoch=1000, norm_data=True):
    rec_loss_train = [None]*epoch
    rec_loss_test = [None]*epoch

    #normalisation des données 
    if(norm_data):
        train, test = normalize_data(train, test)
    
    for n_iter in range(epoch):
        cumul_loss = 0
        for i in range(len(train)):
            rand_ind = torch.randint(len(train),(1,))
            y_hat = Linear.forward(train[rand_ind], w, b)

            loss = MSE.forward(y_hat, y_train[rand_ind])

        
            ##  TODO:  Calcul du backward (grad_w, grad_b)
            loss.backward()
        
            ##  TODO:  Mise à jour des paramètres du modèle
            w.data = w.data - epsilon * w.grad.data
            b.data = b.data - epsilon * b.grad.data
            
            w.grad.data.zero_()
            b.grad.data.zero_()

            cumul_loss += loss

        #`loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', cumul_loss/len(train), n_iter)
            
        # Sortie directe
        print(f"Itérations {n_iter}: loss {cumul_loss/len(train)}")
        rec_loss_train[n_iter]= cumul_loss/len(train)

        #Evalute loss in test
        with torch.no_grad():
            y_hat_test = Linear.forward(test, w, b)
            loss_test = MSE.forward(y_hat_test, y_test)/len(test)
            writer.add_scalar('Loss/test', loss_test, n_iter)
            rec_loss_test[n_iter]=loss_test

    return rec_loss_train, rec_loss_test


def descente_grad_mini_batch(train, test, epsilon=0.01, epoch=1000, batch_size=32, norm_data=True):
    if(norm_data):
        train, test = normalize_data(train, test)
    
    rec_loss_train = [None]*epoch
    rec_loss_test = [None]*epoch
    rand_ind = torch.randperm(len(train))
    nb_batch = len(train)//batch_size +1
      
    for n_iter in range(epoch):
        cumul_loss = 0
        for ind_batch in range(nb_batch):
            
            batch_x = train[rand_ind[ind_batch*batch_size:(ind_batch+1)*batch_size]]
            
            y_hat = Linear.forward(batch_x, w, b)

            loss = MSE.forward(y_hat, y_train[rand_ind[ind_batch*batch_size:(ind_batch+1)*batch_size]])

    
            ##  TODO:  Calcul du backward (grad_w, grad_b)
            loss.backward()
        
            ##  TODO:  Mise à jour des paramètres du modèle
            w.data = w.data - epsilon * w.grad.data
            b.data = b.data - epsilon * b.grad.data
            
            w.grad.data.zero_()
            b.grad.data.zero_()
            cumul_loss += loss

        #`loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', cumul_loss/len(train), n_iter)
            
        # Sortie directe
        print(f"Itérations {n_iter}: loss {cumul_loss/len(train)}")
        rec_loss_train[n_iter]= cumul_loss/len(train)

        #Evalute loss in test
        with torch.no_grad():
            y_hat_test = Linear.forward(test, w, b)
            loss_test = MSE.forward(y_hat_test, y_test)/len(test)
            writer.add_scalar('Loss/test', loss_test, n_iter)
            rec_loss_test[n_iter]=loss_test

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


    
#Init param
w_opt = torch.nn.Parameter(torch.randn(datax.shape[1], datay.shape[1]))
b_opt = torch.nn.Parameter(torch.randn(datay.shape[1]))


#Mini batch avec l'implémentation de l'optimiser    
def mini_batch_optimiser(epsilon=0.01, epoch=1000, batch_size=32, norm_data=True):
    optim = torch.optim.SGD(params=[w_opt, b_opt], lr=epsilon)
    #optim.zero_grad()
    
    if(norm_data):
        train, test = normalize_data(x_train, x_test)
    
    rec_loss_train = [None]*epoch
    rec_loss_test = [None]*epoch
    rand_ind = torch.randperm(len(train))
    nb_batch = len(train)//batch_size +1
      
    for n_iter in range(epoch):
        cumul_loss = 0
        for ind_batch in range(nb_batch):
            
            batch_x = train[rand_ind[ind_batch*batch_size:(ind_batch+1)*batch_size]]
            
            y_hat = Linear.forward(batch_x, w_opt, b_opt)

            loss = MSE.forward(y_hat, y_train[rand_ind[ind_batch*batch_size:(ind_batch+1)*batch_size]])

    
            ##  TODO:  Calcul du backward (grad_w, grad_b)
            loss.backward()
        
            ##  TODO:  Mise à jour des paramètres du modèle
            optim.step()

            ## Reinitialisation du gradient
            optim.zero_grad()
            
            cumul_loss += loss

        #`loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', cumul_loss, n_iter)
            
        # Sortie directe
        print(f"Itérations {n_iter}: loss {cumul_loss}")
        rec_loss_train[n_iter]= cumul_loss

        #Evalute loss in test
        with torch.no_grad():
            y_hat_test = Linear.forward(test, w_opt, b_opt)
            loss_test = MSE.forward(y_hat_test, y_test)
            writer.add_scalar('Loss/test', loss_test, n_iter)
            rec_loss_test[n_iter]=loss_test

    return rec_loss_train, rec_loss_test


#loss_train, loss_test = descente_grad_batch(x_train, x_test, epsilon=1e-4, epoch=1000, norm_data=True)

loss_train, loss_test = descente_grad_stochastic(x_train, x_test, epsilon=1e-6, epoch=1000)

#loss_train, loss_test = descente_grad_mini_batch(x_train, x_test, epsilon=1e-4, epoch=1000, norm_data=True)

#loss_train, loss_test = mini_batch_optimiser(epsilon=1e-3, epoch=100)

plot(loss_train, name_fig='batch_loss_train')
plot(loss_test, name_fig='batch_loss_test')
