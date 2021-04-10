
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import Flatten

import matplotlib.pyplot as plt

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

# choose a number of time steps; here we use three prior time steps of each of
# the two input time series to predict two time steps of the output time series
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)

# summarize the data
for i in range(len(X)):
	print(X[i], y[i])

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define model (SimpleRNN)
model = Sequential()
model.add(SimpleRNN(300, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(TimeDistributed(Dense(n_steps_out, activation='relu')))
model.summary()
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(X, y, epochs=200, batch_size=n_steps_in, verbose=0)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.show()

# demonstrate prediction
x_input = array([[70, 75], [80, 85], [90, 95]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=1)
print(yhat.shape)
print(yhat[0][2])

# Note: Your results may vary given the stochastic nature of the algorithm or
# evaluation procedure, or differences in numerical precision. Consider running
# the example a few times and compare the average outcome.
