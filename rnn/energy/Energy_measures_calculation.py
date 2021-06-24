#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from time import time
import pickle
import sys

from typing import Optional, List

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler


# In[2]:


# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

# multivariate multi-step data preparation
from numpy import array
from numpy.linalg import norm
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


# In[3]:


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



df_pure = pd.read_csv('Energy_reduced_aggr_v2.csv', delimiter=",")

#Remove column "rv2" equal to column "rv1"
df = df_pure.drop(['rv2'],axis=1)



date_col = 1 #number of columns with time stamp
n_var = len(df.columns) - date_col
n_steps_in = 72

#Forecast horizon 
h = 48
# Set value of m to calculate MASE
m = 24


# Load model
model = keras.models.load_model('Energy_aggr_all_lstm_100')


## Training and test set preparation
train_set = array(df.iloc[0:(len(df)-h-n_steps_in),date_col:len(df.columns)])

#Normalization
scaler = StandardScaler()
train_set_norm = scaler.fit_transform(train_set)

test_set = array(df.iloc[(len(df)-h-n_steps_in):len(df),date_col:len(df.columns)])
test_set_norm = scaler.transform(test_set)

# convert into input/output
X_train, y_train = split_sequences(train_set_norm, n_steps_in, h)
X_test = test_set_norm[0:(len(test_set_norm)-h),:]
y_test = test_set_norm[(len(test_set_norm)-h):len(test_set_norm),:]


#Prediction
X_test = X_test.reshape(1,n_steps_in,n_var)
yhat = model.predict(X_test, verbose=0)


#Measures calculation
measures = np.empty((n_var+2,2))
index = []
yhat = yhat.reshape(h, n_var)
df_norm = np.vstack((train_set_norm,test_set_norm))
N = df_norm.shape[0]

for j in range(0,n_var):
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in range(0,h):
        diff = abs(y_test[i,j]-yhat[i,j])
        sum_1 += diff/(abs(y_test[i,j])+abs(yhat[i,j]))
        sum_2 += diff
        
    for i in range(m+1,N):
        sum_3 += abs(df_norm[i,j]-df_norm[i-m,j])
    
    measures[j,0] = 2/h*sum_1*100
    measures[j,1] = (N-m)/h*sum_2/sum_3
    index.append(df.columns[j+date_col])
    
index.append('mean')
measures[n_var,0] = np.mean(measures[0:(n_var),0])
measures[n_var,1] = np.mean(measures[0:(n_var),1])
index.append('std')
measures[n_var+1,0] = np.std(measures[0:(n_var),0])
measures[n_var+1,1] = np.std(measures[0:(n_var),1])

measures_Energy_all = pd.DataFrame(measures,index=index,columns=['sMAPE','MASE'])
measures_Energy_all


sMAPE_vect = 0
MASE_N_vect = 0
for i in range(0,h):
    sMAPE_vect += norm(y_test[i,:]-yhat[i,:],1)/(norm(y_test[i,:],1)+norm(yhat[i,:],1))
    MASE_N_vect += norm(y_test[i,:]-yhat[i,:],1)

MASE_D_vect = 0
for i in range(m+1,N):
    MASE_D_vect += norm(df_norm[i,:]-df_norm[i-m,:],1)

sMAPE_vect = sMAPE_vect*2/h*100
MASE_vect = (N-m)/h*MASE_N_vect/MASE_D_vect
    
measures_vect = np.array([[sMAPE_vect,MASE_vect]])
measures_vect = pd.DataFrame(measures_vect,index=['Energy all'], columns=['sMAPE','MASE'])
measures_vect




##DROP CASE

#Remove variable with correlation index>0.9
df = df_pure.drop(['T9','T_out','rv2'], axis = 1)
date_col = 1 #number of columns with time stamp
n_var = len(df.columns) - date_col


# Load model
model = keras.models.load_model('Energy_aggr_drop_lstm_100')


## Training and test set preparation
train_set = array(df.iloc[0:(len(df)-h-n_steps_in),date_col:len(df.columns)])

#Normalization
scaler = StandardScaler()
train_set_norm = scaler.fit_transform(train_set)

test_set = array(df.iloc[(len(df)-h-n_steps_in):len(df),date_col:len(df.columns)])
test_set_norm = scaler.transform(test_set)

# convert into input/output
X_train, y_train = split_sequences(train_set_norm, n_steps_in, h)
X_test = test_set_norm[0:(len(test_set_norm)-h),:]
y_test = test_set_norm[(len(test_set_norm)-h):len(test_set_norm),:]


#Prediction
X_test = X_test.reshape(1,n_steps_in,n_var)
yhat = model.predict(X_test, verbose=0)


#Measures calculation
measures = np.empty((n_var+2,2))
index = []
yhat = yhat.reshape(h, n_var)
df_norm = np.vstack((train_set_norm,test_set_norm))
N = df_norm.shape[0]

for j in range(0,n_var):
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in range(0,h):
        diff = abs(y_test[i,j]-yhat[i,j])
        sum_1 += diff/(abs(y_test[i,j])+abs(yhat[i,j]))
        sum_2 += diff
        
    for i in range(m+1,N):
        sum_3 += abs(df_norm[i,j]-df_norm[i-m,j])
    
    measures[j,0] = 2/h*sum_1*100
    measures[j,1] = (N-m)/h*sum_2/sum_3
    index.append(df.columns[j+date_col])
    
index.append('mean')
measures[n_var,0] = np.mean(measures[0:(n_var),0])
measures[n_var,1] = np.mean(measures[0:(n_var),1])
index.append('std')
measures[n_var+1,0] = np.std(measures[0:(n_var),0])
measures[n_var+1,1] = np.std(measures[0:(n_var),1])

measures_Energy_all = pd.DataFrame(measures,index=index,columns=['sMAPE','MASE'])
measures_Energy_all


sMAPE_vect = 0
MASE_N_vect = 0
for i in range(0,h):
    sMAPE_vect += norm(y_test[i,:]-yhat[i,:],1)/(norm(y_test[i,:],1)+norm(yhat[i,:],1))
    MASE_N_vect += norm(y_test[i,:]-yhat[i,:],1)

MASE_D_vect = 0
for i in range(m+1,N):
    MASE_D_vect += norm(df_norm[i,:]-df_norm[i-m,:],1)

sMAPE_vect = sMAPE_vect*2/h*100
MASE_vect = (N-m)/h*MASE_N_vect/MASE_D_vect
    
measures_vect = np.array([[sMAPE_vect,MASE_vect]])
measures_vect = pd.DataFrame(measures_vect,index=['Energy drop'], columns=['sMAPE','MASE'])
measures_vect


