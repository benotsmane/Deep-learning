from utils import read_temps, RNN, device, MonDataset, ConstructBatch, DataFile, State
from exo3 import RNN_forecast
import torch
import torch.nn as nn
from pathlib import Path
import os



model_ville1 = Path("model_ville1.pch")
model_ville2 = Path("model_ville2.pch")
model_all_cities = Path("model_all_cities.pch")
model_all_cities_random = Path("model_all_cities_seq_random.pch")

with model_ville1.open("rb") as fp:
    state1 = torch.load(fp)

with model_ville2.open("rb") as fp:
    state2 = torch.load(fp)


with model_all_cities.open("rb") as fp:
    state_all = torch.load(fp)

with model_all_cities_random.open("rb") as fp:
    state_all_random = torch.load(fp)


data_train = read_temps('../../data/tempAMAL_train.csv')
data_test = read_temps('../../data/tempAMAL_test.csv')

train = data_train.narrow(1,0,10)
test = data_test.narrow(1,0,10)

seq_length = 200
len_forecast = 1


start_seq = torch.randint(len(test)-seq_length-len_forecast,(1,))

seq_city1 = torch.unsqueeze(test.narrow(1, 0, 1).narrow(0, start_seq[0], seq_length), 1)
seq_city2 = torch.unsqueeze(test.narrow(1, 1, 1).narrow(0, start_seq[0], seq_length), 1)

seq_all = torch.unsqueeze(test.narrow(1, 0, 10).narrow(0, start_seq[0], seq_length), 1)


#Normalise data
Max1 = torch.max(train[:,0]) 
Min1 = torch.min(train[:,0])
seq_city1_norm = (seq_city1 - Min1)/(Max1-Min1)
pred_city1 = state1.model.forecast(seq_city1_norm, len_forecast)
pred_city1 = pred_city1 * (Max1-Min1) + Min1
                 
print('real temp city 1 : \n' ,test[start_seq+seq_length:start_seq+seq_length+len_forecast, 0].flatten())
print('prediction temp city 1 : \n' ,pred_city1.flatten())


#Normalise data
Max2 = torch.max(train[:,1]) 
Min2 = torch.min(train[:,1])
seq_city2_norm = (seq_city2 - Min2)/(Max2-Min2)
pred_city2 = state2.model.forecast(seq_city2_norm, len_forecast)
pred_city2 = pred_city2 * (Max2-Min2) + Min2
                 
print('real temp city 2 : \n' ,test[start_seq+seq_length:start_seq+seq_length+len_forecast, 1].flatten())
print('prediction temp city 2 : \n' ,pred_city2.flatten())


#Normalise data
Max = torch.max(train) 
Min = torch.min(train)

seq_all_norm = (seq_all - Min)/(Max-Min)
pred_all = state_all.model.forecast(seq_all_norm, len_forecast)
pred_all = pred_all * (Max-Min) + Min
print('real temp city 1 \n' ,test[start_seq+seq_length:start_seq+seq_length+len_forecast, 0].flatten())
print('prediction temp  : \n' ,pred_all.narrow(2,0,1).flatten())

