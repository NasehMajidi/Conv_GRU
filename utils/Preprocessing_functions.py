# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:42:17 2023

@author: Majidi
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import RobustScaler , MinMaxScaler
from sklearn.linear_model import LinearRegression

def scale(data, in_col , scaler):
  out_scaler = scaler.fit(data['train'][in_col].values.reshape(-1,1))
  train_value= out_scaler.transform(data['train'][in_col].values.reshape(-1,1))
  val_value = out_scaler.transform(data['val'][in_col].values.reshape(-1,1))
  test_value = out_scaler.transform(data['test'][in_col].values.reshape(-1,1))
  return train_value , val_value , test_value , out_scaler

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) 

def create_dataset(X,Y,data_mode , time_step=1 , forward_step = 1 ):
    Xs, Ys = [], []
    for i in range(len(X) - time_step - forward_step):
        x = X.iloc[i:(i + time_step)].values
        Xs.append(x)
        if data_mode == "slope":
          Ys.append(Y.iloc[i + time_step -1: i + time_step + forward_step -1 ])
        if data_mode == "single point":
          Ys.append(Y.iloc[i + time_step + forward_step  ])
    if np.array(Xs).ndim == 2 :
      return np.array(Xs).reshape(np.array(Xs).shape[0],np.array(Xs).shape[1],1) , np.array(Ys)
    else:
      return np.array(Xs), np.array(Ys)
  
def coef_maker(x , y , model ):
    coefs = np.zeros((y.shape[0] , 2))
    for i in range(y.shape[0]):
        model.fit(x,y[i])
        a , _ = model.coef_[0] , model.intercept_
        new_y = a*x + y[i][0]
        sigma = mean_absolute_percentage_error(new_y.reshape(-1) , y[i].reshape(-1))
        coefs[i] = a , sigma
    return coefs

def create_coef_data(y):
  coef = dict()
  x = np.arange(y['train'].shape[1]).reshape(-1, 1)
  linear_model =LinearRegression()
  for key in y.keys():
    coefs_out = coef_maker(x , y[key] , linear_model)
    coef[key] = {'mean': coefs_out[:,0],
                    'sigma': coefs_out[:,1]}
  return coef

def coef_scaler(coef ,mean_scaler, sigma_scaler ):
  for key in coef.keys():
    coef[key]['mean'] = coef[key]['mean']/mean_scaler
    coef[key]['sigma'] = coef[key]['sigma']/sigma_scaler

  return coef

