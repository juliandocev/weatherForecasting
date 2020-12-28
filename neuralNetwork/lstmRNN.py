import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout

train_in = np.load('../trainingData/train_in.npy')
train_out = np.load('../trainingData/train_out.npy')
print(train_in)
print(train_out)
neural_network = Sequential()

# create layers
neural_network.add(Bidirectional(LSTM(units=30, return_sequences=True, input_shape=(train_in.shape[1], 1))))
neural_network.add(Dropout(0.2))
neural_network.add(LSTM(units=30, return_sequences=True))
neural_network.add(Dropout(0.2))
neural_network.add(LSTM(units=30, return_sequences=True))
neural_network.add(Dropout(0.2))
neural_network.add(LSTM(units=30))
neural_network.add(Dropout(0.2))
neural_network.add(Dense(units=train_out.shape[1], activation='linear'))

# compile network
neural_network.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])

# teach me master
neural_network.fit(train_in, train_out, epochs=100, batch_size=128)
