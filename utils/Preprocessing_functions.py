# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:42:17 2023

@author: Majidi
"""

###############################################################################
################################### Imports ###################################
###############################################################################
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import RobustScaler , MinMaxScaler
from sklearn.linear_model import LinearRegression

###############################################################################
################################## Functions ##################################
###############################################################################

def scale(data, in_col , scaler):
  """
  Goal: scale the input data using scalers, such as normalization and standardization
  
  Input:
        - data: unscaled data (pandas data frame)
        - in_col: the name of column in data frame which is sclaed
        - scalor: the scalor function
          
  Output:
        - train_value: scaled training data
        - val_value: scaled validation data
        - test_value: scaled test data
        - out_scaler: the scaler cofiguration
  """
  out_scaler = scaler.fit(data['train'][in_col].values.reshape(-1,1))
  train_value= out_scaler.transform(data['train'][in_col].values.reshape(-1,1))
  val_value = out_scaler.transform(data['val'][in_col].values.reshape(-1,1))
  test_value = out_scaler.transform(data['test'][in_col].values.reshape(-1,1))
  return train_value , val_value , test_value , out_scaler

def mean_absolute_percentage_error(y_true, y_pred):
  """
  Goal: calculate the mean absolute percenatage between the inputs
  
  Input:
        - y_true: the groundtruth value
        - y_pred: the predicted value
  
  Output: mean absolute percentage
  """
    return np.mean(np.abs((y_true - y_pred) / y_true)) 

def create_dataset(X,Y,data_mode , time_step=1 , forward_step = 1 ):
  """
  Goal: prepare time series dataset
  
  Input:
        - X: input data
        - Y: label data
        - time_step: the length of X
        - forward_step: the length of Y
  
  Output:
        - Xs: the array of X
        - Ys: the array of Y
  """
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
  """
  Goal: calcualte mean and sigma
  
  Input:
        - x: [1,2,3, ..., forward_step]
        - y: future values of time series based on which sigma and mean are calculated
        - model: the linear model calculating mean (slope) of y
        
  Output:
        - coefs: mean and sigma
  """
    coefs = np.zeros((y.shape[0] , 2))
    for i in range(y.shape[0]):
        model.fit(x,y[i])
        a , _ = model.coef_[0] , model.intercept_
        new_y = a*x + y[i][0]
        sigma = mean_absolute_percentage_error(new_y.reshape(-1) , y[i].reshape(-1))
        coefs[i] = a , sigma
    return coefs

def create_coef_data(y):
  """
  Gaol: Implement a coef dataset
  
  Input:
        - y: the future values in time series
  
  Output:
        - coef: a dictionary containing "mean" and "sigma"
  """
  coef = dict()
  x = np.arange(y['train'].shape[1]).reshape(-1, 1)
  linear_model =LinearRegression()
  for key in y.keys():
    coefs_out = coef_maker(x , y[key] , linear_model)
    coef[key] = {'mean': coefs_out[:,0],
                    'sigma': coefs_out[:,1]}
  return coef

def coef_scaler(coef ,mean_scaler, sigma_scaler ):
  """
  Gaol: scale the coefs' values
  
  Input:
        - coef: unscaled coef values
        - mean_scaler: the division factor of mean
        - sigma_scaler: the division factor of sigma
  
  Output:
        - scaled coef values
  """
  for key in coef.keys():
    coef[key]['mean'] = coef[key]['mean']/mean_scaler
    coef[key]['sigma'] = coef[key]['sigma']/sigma_scaler

  return coef

