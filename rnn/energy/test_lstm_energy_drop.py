#!/usr/bin/env python
# coding: utf-8

import os
import time
import pickle
import sys
import argparse

from typing import Optional, List

import pandas as pd
import numpy as np
print(np.__version__)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import tensorflow as tf
print(tf.__version__)
from tensorflow import keras

#import joblib

# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

# multivariate multi-step data preparation
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

parser = argparse.ArgumentParser()
parser.add_argument('--extras', help='(absolute) path to keras-extras')
parser.add_argument('--gpus', help='number of GPUs')
parser.print_help()
args = parser.parse_args()
#sys.path.append(args.extras)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
ngpus = int(args.gpus)
print("Using %i GPUs" %ngpus)

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



df_pure = pd.read_csv('energydata_complete.csv', delimiter=",")

df_pure.head()


#Remove column "rv2" equal to column "rv1"
df = df_pure.drop(['rv2'],axis=1)
df.head()

date_col = 1 #number of columns with time stamp
n_var = len(df.columns) - date_col
n_steps_in = 72*6

#Forecast horizon
h = 48*6
# Set value of m to calculate MASE
m = 24*6


#Remove variable with correlation index>0.9
df = df_pure.drop(['T9','T_out','rv2'], axis = 1)
date_col = 1 #number of columns with time stamp
n_var = len(df.columns) - date_col

train_set = array(df.iloc[0:(len(df)-h-n_steps_in+1),date_col:len(df.columns)])

# convert into input/output
X_train, y_train = split_sequences(train_set, n_steps_in, h)
X_test = array(df.iloc[(len(df)-h-n_steps_in):(len(df)-h),date_col:len(df.columns)])
y_test = array(df.iloc[(len(df)-h):len(df),date_col:len(df.columns)])


start_time = time.time()

model = keras.models.load_model('Energy_drop_lstm_100_gpu.sav')
model.summary()

X_test = X_test.reshape(1,n_steps_in,n_var)
yhat = model.predict(X_test, verbose=0)


measures = np.empty((n_var+2,2))
index = []
yhat = yhat.reshape(h, n_var)

for j in range(0,n_var):
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in range(0,h):
        diff = abs(y_test[i,j]-yhat[i,j])
        sum_1 += diff/(abs(y_test[i,j])+abs(yhat[i,j]))
        sum_2 += diff

    for i in range(m+1,len(df)):
        sum_3 += abs(df.iloc[i,j+date_col]-df.iloc[i-m,j+date_col])

    measures[j,0] = 2/h*sum_1*100
    measures[j,1] = (len(df)-m)/h*sum_2/sum_3
    index.append(df.columns[j+date_col])

index.append('mean')
measures[n_var,0] = np.mean(measures[0:(n_var),0])
measures[n_var,1] = np.mean(measures[0:(n_var),1])
index.append('std')
measures[n_var+1,0] = np.std(measures[0:(n_var),0])
measures[n_var+1,1] = np.std(measures[0:(n_var),1])

measures_FED_drop = pd.DataFrame(measures,index=index,columns=['sMAPE','MASE'])
print(measures_FED_drop)

print("--- test energy drop %s seconds ---" % (time.time() - start_time))
