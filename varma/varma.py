
# VARMA example
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random

# contrived dataset with dependency
data = list()

for i in range(10):
    v1 = i * 10
    v2 = v1 + 5
    v3 = v1 + v2
    row = [float(v1), float(v2), float(v3)]
    data.append(row)

# fit model
model = VARMAX(data, order=(1,1))
model_fit = model.fit(disp=False)

# make prediction
yhat = model_fit.forecast(steps=1)
print(yhat)
