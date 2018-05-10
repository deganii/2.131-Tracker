import glob

import numpy as np
import csv
import numpy_indexed as npi



data_dim = 2 # 11
timesteps = 300
num_classes = 2

# read in CSV for E.Coli
# read in CSV for Rhodo
root_path = 'C:\\dev\\courses\\2.131 - Advanced Instrumentation\\autocorrelation\\'
ecoli_path = 'ecoli_aut\\'
rhodo_path = 'rhodo_aut\\'
SAMPLE_LEN = 300

def load_from_path(p):
    files = glob.glob(root_path + p + '*.csv')
    dataset = np.zeros(shape=(len(files),SAMPLE_LEN,data_dim))
    for idx, f in enumerate(files):
        dataset[idx] = np.genfromtxt(f,delimiter=',')[:,5:7]
        # subtract the first element from the rest
        dataset[idx] = dataset[idx] - dataset[idx][0]
    return dataset

rhodo = load_from_path(rhodo_path)
ecoli = load_from_path(ecoli_path)

rhodo_labels = np.zeros(rhodo.shape[0])
ecoli_labels = np.ones(ecoli.shape[0])

# split both into training and test
data = np.concatenate((rhodo,ecoli))

# data = data /data.sum(axis=1, keepdims=True)
# data_range = 2*(data - np.max(data))/-np.ptp(data)-1
# data_mean = np.mean(data)

data_min = data.min()
data_max = data.max()

data = 2* (data - data_min) / (data_max - data_min) - 1

# import sklearn.preprocessing as sk
# scaler = sk.MinMaxScaler(feature_range=(-1.0, 1.0), ax)
# scaler = scaler.fit(data)
# data = scaler.transform(data)


labels = np.concatenate((rhodo_labels,ecoli_labels))
# num_labels = label_values.shape[0]


import keras.utils
labels = keras.utils.to_categorical(labels,num_classes)
# labels = np.zeros((num_labels, num_classes))
# labels[np.arange(num_labels), label_values] = 1


split = int(0.7 * data.shape[0])
indices = np.random.permutation(data.shape[0])
training_idx, test_idx = indices[:split], indices[split:]
train,train_labels = data[training_idx,...], labels[training_idx,...]
test, test_labels = data[test_idx,...], labels[test_idx,...]



# from numpy import genfromtxt
# ecoli = genfromtxt(root_path + 'Ecoli-HangingDrop-20x-Dilution-1.MOV-20Trackers.csv', delimiter=',')
# rhodo = genfromtxt(root_path + 'Ecoli-HangingDrop-20x-Dilution-1.MOV-20Trackers.csv', delimiter=',')
#
# # delete the first row
# ecoli = np.delete(ecoli, 0,0)
# rhodo = np.delete(rhodo, 0,0)
#
# SAMPLE_LENGTH = 300
# NUM_TRACKERS = 20
# SN = SAMPLE_LENGTH * NUM_TRACKERS
#
# # make sure the length is exactly divisible by sample lengnth
# new_ecoli_len = int(np.floor(ecoli.shape[0] / (SAMPLE_LENGTH * NUM_TRACKERS)) * SAMPLE_LENGTH * NUM_TRACKERS)
# new_rhodo_len = int(np.floor(rhodo.shape[0] / (SAMPLE_LENGTH * NUM_TRACKERS)) * SAMPLE_LENGTH * NUM_TRACKERS)
#
# num_ecoli_samples = int(new_ecoli_len / SAMPLE_LENGTH)
# num_rhodo_samples = int(new_rhodo_len / SAMPLE_LENGTH)
#
# ecoli = np.delete(ecoli, 0,0)
# rhodo = np.delete(rhodo, 0,0)
#
# ecoli = np.delete(ecoli, np.s_[new_ecoli_len:ecoli.shape[0]], axis=0)
# rhodo = np.delete(rhodo, np.s_[new_rhodo_len:rhodo.shape[0]], axis=0)
#
# ecoli_tracks = npi.group_by(ecoli[:, 0]).split(ecoli)
# rhodo_tracks = npi.group_by(rhodo[:, 0]).split(rhodo)
#
# ecoli_input = np.array(num_ecoli_samples, SAMPLE_LENGTH, 5)
# rhodo_input = np.array(num_rhodo_samples, SAMPLE_LENGTH, 5)
#
# for i in range(20):
#     tracks = ecoli_tracks[i].split()
#     ecoli_input[] =
#
# ecoli.reshape(20, int(new_ecoli_len / SN), SAMPLE_LENGTH, 5)
# rhodo_input = rhodo.reshape(20, int(new_rhodo_len / SN), SAMPLE_LENGTH, 5)
#

# break into tracks, then break each track into a fixed length sequence



# https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa


# load Idaly's files



# add labels

# shuffle

# training / test split


from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import CSVLogger

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(train.shape[1], train.shape[2])))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32

model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()

csv_logger = CSVLogger(root_path+'training.csv',
                       append=True, separator=',')

hist  = model.fit(train, train_labels,
          batch_size=64, epochs=100,
          validation_split=0.2, callbacks=[csv_logger])

model.save(root_path + 'lstm.hdf5')

score = model.evaluate(test, test_labels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
