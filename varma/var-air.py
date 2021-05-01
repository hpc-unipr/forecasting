
# VAR with Air Pollution dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read the data
df = pd.read_csv("../datasets/AirQualityUCI/AirQualityUCI.csv", sep=';', parse_dates=[['Date', 'Time']])
df = df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1)
#df = df.dropna()
#df = df.replace(-200, np.nan)
df = df.dropna()
print(df.describe())

#check the dtypes
#print(df.dtypes)
df['CO(GT)'] = df['CO(GT)'].str.replace(',', '.').astype(float)
df['C6H6(GT)'] = df['C6H6(GT)'].str.replace(',','.').astype(float)
df['T'] = df['T'].str.replace(',', '.').astype(float)
df['RH'] = df['RH'].str.replace(',', '.').astype(float)
df['AH'] = df['AH'].str.replace(',', '.').astype(float)
print(df.dtypes)
print(df.isnull().sum())
print(df.shape)

# for preparing the data, we need the index to have datetime
df['Date_Time'] = pd.to_datetime(df.Date_Time , format = '%d/%m/%Y %H.%M.%S')
data = df.drop(['Date_Time'], axis=1)
data.index = df.Date_Time

#missing value treatment
cols = data.columns
for j in cols:
    for i in range(0,len(data)):
       if data[j][i] == -200:
           data[j][i] = data[j][i-1]

#checking stationarity
from statsmodels.tsa.vector_ar.vecm import coint_johansen
#since the test works for only 12 variables, I have randomly dropped
#in the next iteration, I would drop another and check the eigenvalues
johan_test_temp = data.drop(['CO(GT)'], axis=1)
print(coint_johansen(johan_test_temp,-1,1).eig)

#creating the train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.endog, steps=len(valid))
#print(prediction)

#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,13):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]
#check rmse
#for i in cols:
    #print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid[i])))

# After the testing on validation set, lets fit the model on the complete dataset
#make final predictions
model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.endog, steps=1)
print(yhat)
