# -*- coding: utf-8 -*-
"""
Conductance state binary classifier train

Based on Deep Channel

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

def mcor(y_true, y_pred):
    # Matthews correlation (binary case - simplified)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fn) * (tn + fn)) # Corrected denominator for binary case

    return numerator / (denominator + K.epsilon())


def make_roc(true, predicted): # Modified and color-error fixed

    fpr, tpr, _ = roc_curve(true, predicted)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', # Hardcoded color
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_binary.png') # Save ROC curve plot
    plt.show()

def step_decay(epoch):
    # Learning rate scheduler object
    initial_lrate = 0.001
    drop = 0.001
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

############# SET UP RUN HERE ####################

batch_size = 256

# --- Load Data for Binary Classification ---
df = pd.read_csv('19d11014-90-labeled.csv') # Changed to 19d11014-90-labeled.csv
dataset = df.values.astype('float64')
timep = dataset[:, 0]
maxer = np.amax(dataset[:, 2])
maxeri = maxer.astype('int')
maxchannels = maxeri
idataset = dataset[:, 2].astype(int)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# --- Binary Classification Target: State 0 vs State > 0 ---
binary_idataset = (idataset > 0).astype(int) # Convert to binary: 0 if 0 channels, 1 if >0 channels
y_train_binary = binary_idataset
maxchannels_binary = 1 # Now binary classification

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
y_train = y_train_binary[:] # Use binary targets
x_train = x_train.reshape((len(x_train), 1))
y_train = y_train.reshape((len(y_train), 1))

print("Type of y_train:", type(y_train))
print("Shape of y_train:", y_train.shape)
print("Type of np.unique(y_train):", type(np.unique(y_train)))
print("np.unique(y_train):", np.unique(y_train)) # Print the actual unique values

# Ensure y_train is a 1D numpy array of integers
y_train_1d = y_train_binary.flatten().astype(int) # Flatten y_train_binary directly

# Ensure unique classes is a 1D numpy array of integers
unique_classes = np.unique(y_train_1d).astype(int)

yy_res = y_train.reshape((len(y_train), 1)) # Reshape original y_train
xx_res, yy_res = shuffle(x_train, yy_res) # Shuffle original x_train and reshaped y_train

# --- PRINT CLASS DISTRIBUTION ---
print("\nClass distribution in training set AFTER SMOTE:")
unique_classes_train, class_counts_train = np.unique(yy_res, return_counts=True)
print(dict(zip(unique_classes_train.flatten(), class_counts_train)))

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
df_val = pd.read_csv('19d11014-90-labeled.csv') # Validation data
data_val = df_val.values.astype('float64')

idataset2 = data_val[:, 2].astype(int)
binary_idataset2 = (idataset2 > 0).astype(int) # Binary validation targets
val_target_binary = binary_idataset2

val_set = data_val[:, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
val_set = scaler.fit_transform(val_set.reshape(-1,1))
val_set = val_set.reshape(len(val_set), 1, 1, 1)
val_target = val_target_binary # Use binary validation targets

# --- PRINT VALIDATION CLASS DISTRIBUTION ---
print("\nClass distribution in validation set:")
unique_classes_val, class_counts_val = np.unique(val_target, return_counts=True)
print(dict(zip(unique_classes_val.flatten(), class_counts_val)))

# --- Model --- 

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

newmodel.add(Dense(1)) # Output layer for binary classification (1 unit)
newmodel.add(Activation('sigmoid')) # Sigmoid activation for binary output

newmodel.compile(loss='binary_crossentropy', # Binary crossentropy loss
                optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False), metrics=[
                    'accuracy', Precision(), Recall(), mcor]) # Removed F1Score and categorical metrics

lrate = LearningRateScheduler(step_decay)

epochers = 15
xx_res_reshaped = xx_res.reshape((xx_res.shape[0], 1, 1, 1)) # Reshape xx_res to 4D: (samples, timestep, input_dim, features)
print("Shape of xx_res_reshaped before fit:", xx_res_reshaped.shape) # Verify shape
print("Shape of val_set before fit:", val_set.shape) # Verify shape of validation data too

history = newmodel.fit(xx_res_reshaped, yy_res, # Use xx_res_reshaped and yy_res
                            epochs=epochers,
                            batch_size=32,
                            validation_data=(val_set, val_target),
                            callbacks=[lrate]
                            )

# prediction for val set
predict_val = newmodel.predict(val_set, batch_size=batch_size)

# prediction for test set
predict = newmodel.predict(in_test, batch_size=batch_size)

# --- EVALUATE TEST SET AT THRESHOLD 0.3 ---
threshold_test = 0.3  # Define threshold for test set evaluation
class_predict_test_thresholded = (predict > threshold_test).astype(int) # Apply 0.3 threshold
class_target_test = target_test
cm_test_thresholded = confusion_matrix(class_target_test, class_predict_test_thresholded) # Corrected line - using *_test variables
tn_test, fp_test, fn_test, tp_test = cm_test_thresholded.ravel()

accuracy_test_threshold = (tp_test + tn_test) / (tp_test + tn_test + fp_test + fn_test)
precision_test_threshold = precision_score(class_target_test, class_predict_test_thresholded) # Corrected line - using *_test variables
recall_test_threshold = recall_score(class_target_test, class_predict_test_thresholded) # Corrected line - using *_test variables

# Calculate MCC manually using confusion matrix components
numerator_mcc_test = (tp_test * tn_test) - (fp_test * fn_test)
denominator_mcc_test = np.sqrt((tp_test + fp_test) * (tp_test + fn_test) * (tn_test + fp_test) * (tn_test + fn_test))
if denominator_mcc_test == 0:
    mcc_test_threshold = 0 # Handle division by zero case
else:
    mcc_test_threshold = numerator_mcc_test / denominator_mcc_test

# --- PRINT CONFUSION MATRIX AND METRICS FOR TEST SET AT THRESHOLD 0.3 ---
print(f"\nConfusion Matrix - Test Set (Threshold {threshold_test:.2f}):")
print(cm_test_thresholded) # Corrected line - printing cm_test_thresholded
print(f"\nTest Set Metrics (Threshold {threshold_test:.2f}):")
print(f"Accuracy:  {accuracy_test_threshold:.4f}")
print(f"Precision: {precision_test_threshold:.4f}")
print(f"Recall:    {recall_test_threshold:.4f}")
print(f"MCC:       {mcc_test_threshold:.4f}")

rnd = 1
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy (Binary Classification)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig(str(rnd)+'acc_binary.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss (Binary Classification)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(str(rnd)+'loss_binary.png')
plt.show()

plotlen = test_size
lenny = 2000

plt.figure(figsize=(30, 6))
plt.subplot(2, 1, 1)
# temp=scaler.inverse_transform(dataset)
plt.plot(xx_res[trainy_size:trainy_size+lenny, 0],
      color='blue', label="raw data")
plt.title("Raw Test Data (Binary Classification) - Threshold 0.3") # Updated title to reflect threshold

plt.subplot(2, 1, 2)
plt.plot(class_target_test[:lenny], color='black', label="Actual State", drawstyle='steps-mid') # Corrected variable name

line, = plt.plot(class_predict_test_thresholded[:lenny], color='red', # Corrected variable name
              label="Predicted State (Threshold 0.3)", drawstyle='steps-mid') # Updated label to reflect threshold
plt.setp(line, linestyle='--')
plt.xlabel('timepoint')
plt.ylabel('current')
plt.savefig(str(rnd)+'data_binary_threshold_0_3.png') # Updated filename to include threshold
plt.legend()
plt.show()

# After training (and after evaluation, if desired):
model_save_path = 'anthrax_binary_model_weights.h5' # Changed filename to .h5 for weights
newmodel.save_weights(model_save_path) # Save only weights, not the full model
print(f"Trained model weights saved to: {model_save_path}")

make_roc(val_target, predict_val) # Using predict_val (corrected)

# --- EVALUATE AT DIFFERENT THRESHOLDS ---
thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7] # Define thresholds to evaluate
threshold_metrics = {} # Dictionary to store metrics for each threshold

print("\nEvaluation Metrics at Different Thresholds (Validation Set):")
print("-------------------------------------------------------")
print("Threshold | Accuracy | Precision | Recall | MCC")
print("-------------------------------------------------------")

for threshold in thresholds_to_test:
    class_predict_val_thresholded = (predict_val > threshold).astype(int) # Apply threshold
    cm_val_thresholded = confusion_matrix(val_target, class_predict_val_thresholded)
    tn, fp, fn, tp = cm_val_thresholded.ravel()

    accuracy_threshold = (tp + tn) / (tp + tn + fp + fn)
    precision_threshold = precision_score(val_target, class_predict_val_thresholded) # Use sklearn's precision_score
    recall_threshold = recall_score(val_target, class_predict_val_thresholded) # Use sklearn's recall_score

    # Calculate MCC manually using confusion matrix components
    numerator_mcc = (tp * tn) - (fp * fn)
    denominator_mcc = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator_mcc == 0:
        mcc_threshold = 0 # Handle division by zero case
    else:
        mcc_threshold = numerator_mcc / denominator_mcc

    threshold_metrics[threshold] = { # Store metrics in dictionary
        'accuracy': accuracy_threshold,
        'precision': precision_threshold,
        'recall': recall_threshold,
        'mcc': mcc_threshold,
        'cm': cm_val_thresholded # Optionally store CM if needed for detailed analysis
    }

    print(f"{threshold:.2f}     | {accuracy_threshold:.4f}  | {precision_threshold:.4f}   | {recall_threshold:.4f} | {mcc_threshold:.4f}") # Print formatted output

print("-------------------------------------------------------")
print("\nConfusion Matrices at Different Thresholds (Validation Set):")
for threshold, metrics in threshold_metrics.items():
    print(f"\nThreshold: {threshold:.2f}")
    print(metrics['cm'])
