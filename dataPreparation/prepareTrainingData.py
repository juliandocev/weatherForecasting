import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Input and output dimentions
input_dim = 30
output_dim = 4

# read CSV
csv_data = pd.read_csv('../trainingData/training.csv', sep='\t')
csv_data = csv_data.drop(columns=['Date time'], axis=1)

# scale data
scale = MinMaxScaler(feature_range=(0, 1))
scaled_data = scale.fit_transform(csv_data)

# create training data
train_in = []
train_out = []

for i in range(0, len(scaled_data) - input_dim - output_dim):
    train_in.append(scaled_data[i: i + input_dim, 0])
    train_out.append(scaled_data[i + input_dim: i + input_dim + output_dim, 0])

train_in = np.array(train_in)
train_out = np.array(train_out)
train_in = np.reshape(train_in, (train_in.shape[0], train_in.shape[1], 1))
print(train_out.shape)
print(train_in.shape)
np.save('../trainingData/train_in', train_in)
np.save('../trainingData/train_out', train_out)

