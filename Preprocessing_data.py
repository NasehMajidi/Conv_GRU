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

###############################################################################
##################################### Main ####################################
###############################################################################

# load the preprocessed data
with open('Dataset/preprocessed_data.pickle', 'wb') as handle:
    pickle.dump(data, handle)

# Mean Sigma Preprocessing
time_step = 32
forward_step = 10
target_col = "btc"
data_mode = "slope"
x_data = dict()
y_data = dict()
for key in data.keys():
    x_data[key] , y_data[key] = create_dataset(data[key][target_col], data[key][target_col],
                                     time_step=time_step , forward_step=forward_step ,
                                     data_mode = data_mode )
print('---------- Preprocessed Data Summary ----------')
for key in data.keys():
    print(f' x: {key} data --------> {x_data[key].shape}')
    print(f' y: {key} data --------> {y_data[key].shape}')
print('-----------------------------------------------')

coef = create_coef_data(y_data)


mean_scaler, sigma_scaler = np.quantile(coef['train']['mean'] , 0.99) , np.quantile(coef['train']['sigma'] , 0.99) 
print('--------- Mean Sigma Scaler Summary ---------')
print(f"Mean Scaler 10: {mean_scaler:.2f}  ,Sigma Scaler 10: {sigma_scaler:.2f}")
print('---------------------------------------------')

coef = coef_scaler(coef, mean_scaler, sigma_scaler)
with open('Dataset/preprocessed_coef.pickle', 'wb') as handle:
    pickle.dump(coef, handle)
    
data_loader = dict()
for key in coef.keys():
    data_list = []
    mean_ = np.reshape(coef[key]['mean'],(coef[key]['mean'].shape[0],1))
    sigma_ = np.reshape(coef[key]['sigma'],(coef[key]['sigma'].shape[0],1))
    mean_sigma = np.concatenate((mean_,sigma_),1)
    for x_ , y_ in zip(x_data[key] , mean_sigma):
        data_list.append((x_,y_))
    data_loader[key] = data_list

print('----- Data loader summary -----')
for key in data_loader.keys():
    print(f"# {key}: {len(data_loader[key])}    ")
print('-------------------------------')

with open('Dataset/preprocessed_dataloader.pickle', 'wb') as handle:
    pickle.dump(data_loader, handle)
