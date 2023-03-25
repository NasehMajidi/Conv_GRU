# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 01:27:12 2023

@author: Majidi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import  DataLoader
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 




class Train:
    def __init__(self, model, device, train_loader, val_loader, optimizer, cost, 
                 l1_lambda, epoch_res, total_epoch, verbose):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.cost = cost
        self.l1_lambda = l1_lambda
        self.epoch_res = epoch_res
        self.total_epoch = total_epoch
        self.verbose = verbose
        
    def loss_acc_cal(self, output , target):
      y_mu = output[: , 0].cpu().detach().numpy()
      y_sigma = output[: , 1].cpu().detach().numpy()
      target_mu = target[:,0].cpu().detach().numpy()
      target_sigma = target[:,1].cpu().detach().numpy()
      mse_slope = round(mean_squared_error(y_mu ,target_mu) , 5)
      mse_sigma = round(mean_squared_error(y_sigma ,target_sigma) , 5)
      error_Slope = len(y_mu) *mse_slope
      a = np.sign(y_mu) 
      b = np.sign(target_mu)
      c = a*b
      true_count = np.count_nonzero(c == 1)
      false_count =  np.count_nonzero(c == -1)
      return true_count ,false_count ,  error_Slope 

    
    def train_step(self):
      self.model.train()
      losses = []
      out_ = []
      cum_true = 0
      cum_false = 0
      cum_mse = 0
      for batch_idx, (data, target) in enumerate(self.train_loader):
        data, target = data.float().to(self.device), target.float().to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        true , false , mse_s = self.loss_acc_cal(output , target)
        cum_true += true
        cum_false += false
        cum_mse += mse_s

        loss = self.cost(output, target).to(self.device)
        if self.l1_lambda :  
          regularization_loss = 0
          for param in self.model.parameters():
              regularization_loss += torch.sum(abs(param))
          loss = loss + self.l1_lambda * regularization_loss
        loss.backward()
        self.optimizer.step()
        losses.append(float(loss))
      epoch_loss = np.array(losses).mean()
      acc = (100 * cum_true) / (cum_true + cum_false)
      cum_mse = cum_mse / (cum_true + cum_false)
      return epoch_loss ,  acc , cum_mse
    
    def val_step(self):
      self.model.eval()
      v_losses = []
      cum_true = 0
      cum_false = 0
      cum_mse = 0
      with torch.no_grad():
        for batch_idx, (data, target) in enumerate(self.val_loader):
          data, target = data.float().to(self.device), target.float().to(self.device)
          #data = data.view(-1, input_channels, seq_length)
          output = self.model(data)
          true , false , mse_s = self.loss_acc_cal(output , target)
          cum_true += true
          cum_false += false
          cum_mse += mse_s

          v_loss = self.cost(output, target).to(self.device)
          if self.l1_lambda :  
            regularization_loss = 0
            for param in self.model.parameters():
                regularization_loss += torch.sum(abs(param))
            v_loss = v_loss + self.l1_lambda * regularization_loss
          v_losses.append(float(v_loss))
      epoch_v_loss = np.array(v_losses).mean()
      acc = (100 * cum_true) / (cum_true + cum_false)
      cum_mse = cum_mse / (cum_true + cum_false)
      return epoch_v_loss ,  acc , cum_mse
    
    def train(self):
      history = dict()
      train_loss_history = []
      val_loss_history = []
      train_mse_history = []
      val_mse_history = []
      train_acc_history = []
      val_acc_history = []
      min_train = 1e5
      min_val = 1e5
      for epoch in range(self.total_epoch):
        t0 = time.time()
        train_loss , train_acc , train_mse = self.train_step()
        val_loss , val_acc , val_mse = self.val_step()
        t1 = time.time()
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        if train_loss < min_train:
          min_train = train_loss
          self.best_train_model = self.model
        if val_loss < min_val:
          min_val = val_loss
          self.best_val_model = self.model

        self.history = {'train':
                      {'loss': train_loss_history,
                      'mse':train_mse_history,
                      'acc':train_acc_history},
                  'val':
                      {'loss': val_loss_history,
                      'mse':val_mse_history,
                      'acc':val_acc_history}}
        if not (epoch) % self.epoch_res and self.verbose:

          print(f'{epoch} / {self.total_epoch}\n--------')
          print('     Train loss : {:.4f}         Validation loss : {:.4f}         Time: {:.2f} Mins'.format(float(train_loss) ,float(val_loss),(t1-t0)/60  ))    
          print('     Train MSE : {:.4f}    Validation MSE: {:.4f}'.format(train_mse , val_mse))
          print('     Train ACC : {:.1f}    Validation ACC: {:.1f}'.format(train_acc , val_acc))


      return self.model , self.history

class ConvGru(nn.Module):
    def __init__(self, kernels  , channels   , gru_dim , lin_dim, dropout , padding = 1 , num_l = 2 , in_channel = 1):
      super(ConvGru, self).__init__()

      self.relu = nn.LeakyReLU()
      self.drop = nn.Dropout(dropout)
      self.convs = nn.ModuleList()
      for i in range(len(kernels)):
        if i == 0:
          self.convs.append(nn.Conv1d(in_channel, channels[i], kernel_size=kernels[i], padding=padding))
        else:
          self.convs.append(nn.Conv1d(channels[i-1], channels[i], kernel_size=kernels[i], padding=padding))
      
    
      
      self.grus = nn.ModuleList()
      for i in range(len(gru_dim)):
        if i == 0:
          self.grus.append(nn.GRU(channels[-1], gru_dim[i], batch_first=True,num_layers= num_l, dropout=dropout))
        else:
          self.grus.append(nn.GRU(gru_dim[i-1], gru_dim[i], batch_first=True, dropout=dropout))

      self.conv_nets = nn.ModuleList()
      for i in range(len(self.convs)):
        self.conv_nets.append(nn.Sequential(self.convs[i] , nn.BatchNorm1d(channels[i]), nn.ReLU(), self.drop))
      
      self.denses = nn.ModuleList()
      for i in range(len(lin_dim)):
        if i == 0:
          self.denses.append(nn.Linear(gru_dim[-1], lin_dim[i]))
        else:
          self.denses.append(nn.Linear(lin_dim[i-1], lin_dim[i]))
      self.denses.append(nn.Linear(lin_dim[-2], lin_dim[-1]))
        
      self.init_weights()

    def init_weights(self):
      for i in range(len(self.convs)):
        torch.nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain('relu'))

      for i in range(len(self.denses)):
        self.denses[i].weight.data.normal_(0.0 , 0.02)
        self.denses[i].bias.data.fill_(0)



    def forward(self, x):
      x = x.transpose(1, 2)
      for i in range(len(self.conv_nets)):
        conv = self.convs[i]
        x = conv(x)
      x = x.permute(0, 2, 1)
      for i in range(len(self.grus)):
        gru = self.grus[i]
        x = gru(x)[0]
      x = x[:, -1, :]
      for i in range(len(self.denses) - 2):
        x = self.drop(self.denses[i](x))
        x = self.relu(x)
      sl = torch.tanh(self.denses[-2](x))
      sigma = torch.sigmoid(self.denses[-1](x))
      return torch.cat((sl , sigma) , 1)


      return out

class PenaltyLoss ():
  def __init__(self ,k = 10,  alpha_penalty = 0.75 , alpha_cl = 0.5 , reduction = "mean"):
    self.alpha_penalty = alpha_penalty
    self.k = k
    self.mse = nn.MSELoss(reduction = reduction)
    self.cl = CustomLoss("mse" , mode = "both" , alpha_cl=alpha_cl)
  def __call__ (self , output , target):
    a = torch.tanh(self.k*output[: , 0])
    b = torch.tanh(self.k*target[: , 0])
    loss_penalty = self.mse(a,b) 
    loss_cl = self.cl(output ,target )

    loss = self.alpha_penalty * loss_cl + (1-self.alpha_penalty) * loss_penalty
    return loss

class CustomLoss():
  def __init__(self,loss_f , beta = 1 , alpha_cl = 0.5 ,reduction='mean' , mode = 'both'):
    self.alpha_cl = alpha_cl
    if loss_f == "mse":
      self.cost = nn.MSELoss(reduction = reduction) 
    if loss_f == "huber":
      self.cost = nn.SmoothL1Loss(beta = beta)
    self.mode = mode
  def __call__(self, output ,  target ):
    if self.mode == "both":
      loss = self.alpha_cl * self.cost(output[: , 0] , target[: , 0]) + (1-self.alpha_cl)*self.cost(output[: , 1] , target[: ,1])
    else:
      loss = self.cost(output[:,0], target)
    return loss

def loader(data, batch_size , shuffle_mode = False, data_len = -1):
  data_loader = dict()
  if data_len == -1:
    for key in data.keys():
      data_loader[key] = DataLoader(data[key], batch_size = batch_size , shuffle=shuffle_mode)
  else:
    for key in data.keys():
      data_loader[key] = DataLoader(data[key][:data_len], batch_size = batch_size , shuffle=shuffle_mode)
  return data_loader