# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:37:34 2023

@author: Majidi
"""
###############################################################################
################################### Imports ###################################
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import RobustScaler , MinMaxScaler
from sklearn.linear_model import LinearRegression
import pickle
from utils.Preprocessing_functions import *

import seaborn as sns
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set(style='whitegrid',font_scale=2.5)
rcParams['figure.figsize'] = 22 , 10
###############################################################################
#################################### Main #####################################
###############################################################################

# load btc data
btc = pd.read_csv(
  r"Dataset/bitcoin_1d.csv",  
  parse_dates=['Date'],
  index_col="Date"
)

dataset = pd.DataFrame()
dataset["btc"] = btc["Adj Close"].copy()

dataset.describe()

# train-validation-test split
train_size = int(len(dataset) * 0.80)
val_size =  int(len(dataset) * 0.10)
test_size = len(dataset) - train_size - val_size
data = dict()
data['train'] , data['val'] , data['test'] = dataset.iloc[0:train_size].copy(), dataset.iloc[train_size:train_size + val_size].copy() , dataset.iloc[train_size+val_size:len(dataset)].copy()
print('------- Dataset Summary ----------')
print("# dataset   :   ",len(dataset))
print("# train     :   ",len(data['train']))
print("# validation:   ",len(data['val']))
print("# test      :   ",len(data['test']))
print('-----------------------------------')

# calculate the percentage error of price data
for key in data.keys():
    data[key]['pct'] = np.log(data[key]['btc'].pct_change() + 1)
    data[key] = data[key].dropna()

# scale the price data (normalization phase)
target_column = 'pct'
out_col = target_column + "_mm0" 
data['train'][out_col] ,data['val'][out_col], data['test'][out_col], out_scaler = scale(data , target_column , MinMaxScaler())

