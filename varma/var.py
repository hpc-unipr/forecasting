
# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random

# contrived dataset with dependency
data = list()
for i in range(10):
    v1 = i * 10
    v2 = v1 + 5
    v3 = v1 + v2
    row = [v1, v2, v3]
    data.append(row)

# fit model
model = VAR(data)
model_fit = model.fit()

# make prediction
yhat = model_fit.forecast(model_fit.endog, steps=2)
print(yhat)
