# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 01:32:52 2023

@author: Majidi
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
import pickle
from utils.Model_functions import *
from tqdm.notebook import tqdm


with open("Dataset/preprocessed_dataloader.pickle", "rb") as input_file:
    data_loader = pickle.load(input_file)
    
my_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'My device is "{my_device}"')

batch_size = 32 
epochs = 5 + 1
data_len = 320
data_loader2 = loader(data_loader, batch_size , shuffle_mode=True, data_len = data_len)
num_channels_v = [[64 , 64 ]  , [64 , 64 , 32] , [64 , 64 , 32 , 32] , [32 , 32] , [32 , 32 , 16] , [32 , 32 , 16 , 16] , [64 , 64 , 32 , 32 , 32] , [16 , 32 , 32 , 32 , 64]]
lrs = np.logspace(-4 ,-1, 5)
drop = 0.0
j = 0
alpha = 0.5
for lr in tqdm(lrs):
  i = 0
  for num_channels in tqdm(num_channels_v):
    kernel_size = len(num_channels)*[5]
    model = ConvGru(kernels = kernel_size, channels = num_channels , gru_dim = [64] , lin_dim = [128,1], dropout = 0)
    device =my_device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters() ,lr = lr )
    cost_function = PenaltyLoss(k = 10 , alpha_penalty=0.6 , alpha_cl=0.6 )
    initial_t = time.time()
    training_model = Train(model, device,  data_loader2['train'], data_loader2['train'] , optimizer, cost_function, 0 ,
                  10, epochs, False)
    my_model , history = training_model.train()
    end_t = time.time()
    print(f"{j+1:3d}/{len(lrs) * len(num_channels_v) :3d}:")
    print(f"     Model:{i} -->  lr: {lr:.4f}, Duration: {(end_t - initial_t ) / 60:.2f} Mins")
    print(f"     Train ---->  Loss: {min(history['train']['loss']):.4f},  Acc:{max(history['train']['acc']):2f},  MSE: {min(history['train']['mse']):.4f}") 
    print(f"     Val   ---->  Loss: {min(history['val']['loss']):.4f},  Acc:{max(history['val']['acc']):2f},  MSE: {min(history['val']['mse']):.4f}") 
          
    i = i+1
    j = j+1
    

############Tuning##########
batch_size = 32 
epochs = 1 + 1
data_loader2 = loader(data_loader, batch_size , shuffle_mode=True, data_len = data_len)
num_channels_v = [[64 , 64 ]  , [64 , 64 , 32] , [64 , 64 , 32 , 32] , [32 , 32] , [32 , 32 , 16] , [32 , 32 , 16 , 16] , [64 , 64 , 32 , 32 , 32] , [16 , 32 , 32 , 32 , 64]]
l1_regs = np.logspace(-5 , -3 , 100)
regs = np.logspace(-5 , -3 , 100)
lrs = np.logspace(-4 , -2 , 100)
drops = np.arange(0.0 , 0.3 , 0.1)
model_ind = 7
max_range = 60
num_channels= num_channels_v[model_ind]
kernel_size = len(num_channels)*[3]
for i in tqdm(range(max_range)):
  l1_reg = 0
  if i%3 == 0:
    l1_reg = round(np.random.choice(l1_regs) , 5)
  reg = round(np.random.choice(regs) , 5)
  lr = round(np.random.choice(lrs) , 5)
  drop = round(np.random.choice(drops) , 2)

  model = ConvGru(kernels = kernel_size, channels = num_channels , gru_dim = [64] , lin_dim = [128,1], dropout = 0)
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters() ,lr = lr , weight_decay = reg )
  initial_t = time.time()

  training_model = Train(model, device,  data_loader2['train'], data_loader2['val'] , optimizer, cost_function, l1_reg ,
                  10, epochs, False)
  my_model , history = training_model.train()


  end_t = time.time()
  dur = int(end_t - initial_t )

  #print(f'{i}/{max_range} --- > lr: {round(lr , 6)} , reg: {round(reg , 6)} , train: {round(min(train_loss) , 5)}  , val: {round(min(val_loss) , 5)} , Time: {dur} Secs ')
  print(f'{i+1:3}/{max_range:3} --- > lr: {lr:.6f} , reg: {reg:.5f}  , L1 reg:{l1_reg:.5f}  , drop :{drop:.2f}' )
  print(f"      Train ------------>  Loss: {min(history['train']['loss']):.4f},  Acc:{max(history['train']['acc']):2f},  MSE: {min(history['train']['mse']):.4f}") 
  print(f"      Val   ------------>  Loss: {min(history['val']['loss']):.4f},  Acc:{max(history['val']['acc']):2f},  MSE: {min(history['val']['mse']):.4f}") 

