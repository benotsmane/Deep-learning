import optuna
import torch.nn as nn
from main import NN
from main import GradientDescent
from main import *


#PARAMS
epoch=100
BATCH_SIZE=300
dim_input=28*28
dim_hidden_layer= 100
dim_output=10


#data
ds = prepare_dataset("com.lecun.mnist")
train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
test_img, test_labels = ds.test.images.data(), ds.test.labels.data()

train_dataset = MonDataset(train_img, train_labels)
test_dataset = MonDataset(test_img, test_labels)

train_size = len(train_dataset)

indices = torch.randperm(train_size)  
train_subset = Subset(train_dataset, indices[:int(TRAIN_RATIO*train_size)])
val_subset = Subset(train_dataset, indices[int(TRAIN_RATIO*train_size):])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset , batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



def objective(trial):
    ratio_dropout = trial.suggest_discrete_uniform('ratio_dropout', 0.0, 1.0, 0.1)
    normalisation = trial.suggest_categorical('normalisation', ['None', 'batch', 'layer'])

    lr = trial.suggest_categorical('lr', [1e-2, 1e-3, 1e-4])
    lambda1 = trial.suggest_categorical('lambda1', [0, 1e-3, 1e-4, 1e-5])
    lambda2 = trial.suggest_categorical('lambda2', [0, 1e-3, 1e-4, 1e-5])
    
    model = NN(dim_input, dim_output, dim_hidden_layer, ratio_dropout=ratio_dropout, norm=normalisation)
    model = model.to(device)
    optimizer = GradientDescent(model, lr, epoch)
    
    loss, accuracy = optimizer.descente_gradient(train_loader, val_loader, lambda1=lambda1, lambda2=lambda2)

    model.cpu()

    return min(loss[1])

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print(study.best_params)
