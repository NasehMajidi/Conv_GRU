# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 03:18:34 2023

@author: Majidi
"""
###############################################################################
################################### Imports ###################################
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
import pickle
from utils.Model_functions import *

###############################################################################
#################################### Main #####################################
###############################################################################

# load preprocessed data loader
with open("Dataset/preprocessed_dataloader.pickle", "rb") as input_file:
    data_loader = pickle.load(input_file)

# load preprocessed coef
with open("Dataset/preprocessed_coef.pickle", "rb") as input_file:
    coef = pickle.load(input_file)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'My device is "{device}"')

# Train and Test the final model
batch_size = 32 
epochs = 15 + 1
data_loader2 = loader(data_loader, batch_size , shuffle_mode=True)
num_channels_v = [[64 , 64 ]  , [64 , 64 , 32] , [64 , 64 , 32 , 32] , [32 , 32] , [32 , 32 , 16] , [32 , 32 , 16 , 16] , [64 , 64 , 32 , 32 , 32] , [16 , 32 , 32 , 32 , 64]]
model_ind = 7
num_channels= num_channels_v[model_ind]
kernel_size = len(num_channels)*[3]
l1_reg = 0
reg = 0
lr = 1e-4
drop = 0.2
cost_function = PenaltyLoss(k = 10 , alpha_penalty=0.6 , alpha_cl=0.6 )
model = ConvGru(kernels = kernel_size, channels = num_channels , gru_dim = [64] , lin_dim = [128,1], dropout = 0)
model.to(device)
optimizer = torch.optim.Adam(model.parameters() ,lr = lr , weight_decay = reg )
train_loss = []
val_loss = []
initial_t = time.time()

training_model = Train(model, device,  data_loader2['train'], data_loader2['val'] , optimizer, cost_function, l1_reg ,
              10, epochs, True)
my_model , history = training_model.train()

testing_model = Test(my_model, device, data_loader2['test'], coef['test'])
mse_mean , mse_sigma , acc = testing_model.loss_acc()

print('------- Model testing -------')
print(f' MSE of mean: {mse_mean}, MSE of sigma: {mse_sigma}, Acc: {acc}')
