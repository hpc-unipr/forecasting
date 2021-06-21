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


# <font size="5">Energy Dataset</font>

# In[4]:


df_pure = pd.read_csv('Energy_reduced.csv', delimiter=",")

df_pure.head()


# In[5]:


#Remove column "rv2" equal to column "rv1"
df = df_pure.drop(['rv2'],axis=1)
df.head()


# In[6]:


len(df)


# In[7]:


date_col = 1 #number of columns with time stamp
n_var = len(df.columns) - date_col
n_steps_in = 72*6

#Forecast horizon 
h = 48*6
# Set value of m to calculate MASE
m = 24*6


# In[8]:


train_set = array(df.iloc[0:(len(df)-h-n_steps_in+1),date_col:len(df.columns)])

#Normalization
scaler = StandardScaler()
train_set_norm = scaler.fit_transform(train_set)

test_set = array(df.iloc[(len(df)-h-n_steps_in):len(df),date_col:len(df.columns)])
test_set_norm = scaler.transform(test_set)

# convert into input/output
X_train, y_train = split_sequences(train_set_norm, n_steps_in, h)
X_test = test_set_norm[(len(test_set_norm)-h-n_steps_in):(len(test_set_norm)-h),:]
y_test = test_set_norm[(len(test_set_norm)-h):len(test_set_norm),:]


# In[ ]:


# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_var)))
model.add(RepeatVector(h))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_var)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_train, y_train, epochs=100, verbose=0)
model.save('Energy_all_lstm.h5')


# In[ ]:


X_test = X_test.reshape(1,n_steps_in,n_var)
yhat = model.predict(X_test, verbose=0)




# In[ ]:


#Remove variable with correlation index>0.9
df = df_pure.drop(['T9','T_out','rv2'], axis = 1)
date_col = 1 #number of columns with time stamp
n_var = len(df.columns) - date_col


# In[ ]:


train_set = array(df.iloc[0:(len(df)-h-n_steps_in+1),date_col:len(df.columns)])

#Normalization
scaler_drop = StandardScaler()
train_set_norm = scaler_drop.fit_transform(train_set)

test_set = array(df.iloc[(len(df)-h-n_steps_in):len(df),date_col:len(df.columns)])
test_set_norm = scaler_drop.transform(test_set)

# convert into input/output
X_train, y_train = split_sequences(train_set_norm, n_steps_in, h)
X_test = test_set_norm[(len(test_set_norm)-h-n_steps_in):(len(test_set_norm)-h),:]
y_test = test_set_norm[(len(test_set_norm)-h):len(test_set_norm),:]


# In[ ]:


# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_var)))
model.add(RepeatVector(h))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_var)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_train, y_train, epochs=100, verbose=0)
model.save('Energy_drop_lstm.h5')


# In[ ]:


X_test = X_test.reshape(1,n_steps_in,n_var)
yhat = model.predict(X_test, verbose=0)





# In[ ]:





# In[ ]:





# In[ ]:




