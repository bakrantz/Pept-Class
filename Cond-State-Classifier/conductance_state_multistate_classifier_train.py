# -*- coding: utf-8 -*-
"""
Conductance State Multi-state Classifier

Based on Deep-Channel

B. Krantz
"""

# Importing the libraries
import os
import numpy
import time
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score # Import precision_score and recall_score
from sklearn.utils import shuffle
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation, LSTM, BatchNormalization, TimeDistributed, Conv1D, MaxPooling1D
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from tensorflow_addons.metrics import F1Score

'''
############# SET UP RUN HERE ####################
'''

batch_size = 256

# --- Load Data for Three-state Classification ---
df = pd.read_csv('16203009-labeled-filtered-raw-data-window-size-3.csv') # Labeled 3-state data
dataset = df.values.astype('float64')
timep = dataset[:, 0]
maxer = np.amax(dataset[:, 2])
maxstates = maxer.astype('int')
num_states = maxstates + 1 # The number of states is one more than the max states since class 0 is a state  
idataset = dataset[:, 2].astype(int)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# train and test set split and reshape:
train_size = int(len(dataset) * 0.80)
modder = math.floor(train_size/batch_size)
train_size = int(modder*batch_size)
test_size = int(len(dataset) - train_size)
modder = math.floor(test_size/batch_size)
test_size = int(modder*batch_size)

print(f'training set = {train_size}')
print(f'test set = {test_size}')
print(f'total length = {test_size + train_size}')

x_train = dataset[:, 1]
y_train = idataset # Now for multi-state
x_train = x_train.reshape((len(x_train), 1))
y_train = y_train.reshape((len(y_train), 1))

print("Type of y_train:", type(y_train))
print("Shape of y_train:", y_train.shape)
print("Type of np.unique(y_train):", type(np.unique(y_train)))
print("np.unique(y_train):", np.unique(y_train)) # Print the actual unique values

# Ensure y_train is a 1D numpy array of integers
y_train_1d = y_train.flatten().astype(int) # Flatten y_train_binary directly

# Ensure unique classes is a 1D numpy array of integers
unique_classes = np.unique(y_train_1d).astype(int)

yy_res = y_train.reshape((len(y_train), 1)) # Reshape original y_train
yy_res = to_categorical(yy_res, num_classes=num_states) # Now categorical
xx_res, yy_res = shuffle(x_train, yy_res) # Shuffle original x_train and reshaped y_train

# --- PRINT CLASS DISTRIBUTION IN TRAINING SET ---
print("\nClass distribution in training set:")
# Sum along axis=0 to get counts for each class (across all samples)
class_counts_train = np.sum(yy_res, axis=0)
# Assuming classes are 0, 1, 2... use range(num_states) for class labels
class_labels_train = range(num_states)
class_distribution_train = dict(zip(class_labels_train, class_counts_train))
print(class_distribution_train)

trainy_size = int(len(xx_res) * 0.80)
modder = math.floor(trainy_size/batch_size)
trainy_size = int(modder*batch_size)
testy_size = int(len(xx_res) - trainy_size)
modder = math.floor(testy_size/batch_size)
testy_size = int(modder*batch_size)

print('training set= ', trainy_size)
print('test set =', testy_size)
print('total length', testy_size+trainy_size)

in_train, in_test = xx_res[0:trainy_size,
                                    0], xx_res[trainy_size:trainy_size+testy_size, 0]
target_train, target_test = yy_res[0:trainy_size,
                                        :], yy_res[trainy_size:trainy_size+testy_size, :]
in_train = in_train.reshape(len(in_train), 1, 1, 1)
in_test = in_test.reshape(len(in_test), 1, 1, 1)

# --- Validation set ---
df_val = pd.read_csv('16203009-labeled-filtered-raw-data-window-size-3.csv') # Validation data
data_val = df_val.values.astype('float64')

idataset2 = data_val[:, 2].astype(int)
val_target = idataset2

val_set = data_val[:, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
val_set = scaler.fit_transform(val_set.reshape(-1,1))
val_set = val_set.reshape(len(val_set), 1, 1, 1)
val_target = to_categorical(val_target, num_classes=num_states) # Now categorical

# --- PRINT CLASS DISTRIBUTION IN VALIDATION SET --- # Added Class Distribution Printing Block
print("\nClass distribution in validation set:")
# Sum along axis=0 to get counts for each class (across all samples)
class_counts_val = np.sum(val_target, axis=0)
# Assuming classes are 0, 1, 2... use range(num_states) for class labels
class_labels_val = range(num_states)
class_distribution_val = dict(zip(class_labels_val, class_counts_val))
print(class_distribution_val)


# --- Model starts ---

newmodel = Sequential()
timestep = 1
input_dim = 1
newmodel.add(TimeDistributed(Conv1D(filters=64, kernel_size=1,
                                        activation='relu'), input_shape=(None, timestep, input_dim)))
newmodel.add(TimeDistributed(MaxPooling1D(pool_size=1)))
newmodel.add(TimeDistributed(Flatten()))

newmodel.add(LSTM(256, activation='relu', return_sequences=True))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.25))

newmodel.add(LSTM(256, activation='relu', return_sequences=True))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.25))

newmodel.add(LSTM(256, activation='relu'))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.25))

newmodel.add(Dense(num_states, activation='softmax')) # Output layer(s) for multistate classification

newmodel.compile(loss='categorical_crossentropy',  # Categorical crossentropy loss
                 optimizer='adam', # Switched to Adam optimizer
                 metrics=[
                     'accuracy',
                     Precision(name='precision'),
                     Recall(name='recall'),
                     F1Score(num_classes=int(num_states), average='macro')]
                 ) # Removed mcor, kept macro-averaged metrics

epochers = 15
xx_res_reshaped = xx_res.reshape((xx_res.shape[0], 1, 1, 1)) # Reshape xx_res to 4D: (samples, timestep, input_dim, features)
print("Shape of xx_res_reshaped before fit:", xx_res_reshaped.shape) # Verify shape
print("Shape of val_set before fit:", val_set.shape) # Verify shape of validation data too

history = newmodel.fit(xx_res_reshaped, yy_res, # Use xx_res_reshaped and yy_res
                            epochs=epochers,
                            batch_size=32,
                            validation_data=(val_set, val_target)
                            )

# prediction for val set
predict_val = newmodel.predict(val_set, batch_size=batch_size)

# prediction for test set
predict = newmodel.predict(in_test, batch_size=batch_size)

rnd = 1
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig(str(rnd)+'acc.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss (Binary Classification)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(str(rnd)+'loss.png')
plt.show()

plotlen = test_size
lenny = 2000

plt.figure(figsize=(30, 6))
plt.subplot(2, 1, 1)
plt.plot(xx_res[trainy_size:trainy_size+lenny, 0],
      color='blue', label="raw data")
plt.title("Raw Test Data") # Updated title to reflect threshold

plt.subplot(2, 1, 2)
# plt.plot(class_target_test[:lenny], color='black', label="Actual State", drawstyle='steps-mid') # Corrected variable name

# line, = plt.plot(class_predict_test_thresholded[:lenny], color='red', # Corrected variable name
#              label="Predicted State", drawstyle='steps-mid') # Updated label to reflect threshold
# plt.setp(line, linestyle='--')
plt.xlabel('timepoint')
plt.ylabel('current')
plt.savefig(str(rnd) + 'data.png')
plt.legend()
plt.show()

# --- Save weights ---
model_save_path = 'anthrax_three_state_model_weights.h5' # Changed filename to .h5 for weights
newmodel.save_weights(model_save_path) # Save only weights, not the full model
print(f"Trained model weights saved to: {model_save_path}")
