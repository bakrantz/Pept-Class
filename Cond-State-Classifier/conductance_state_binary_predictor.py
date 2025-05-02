# -*- coding: utf-8 -*-
"""
Anthrax Binary Classification Predictor

Based on Deep Channel

B. Krantz
"""
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the Keras libraries and packages
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # Keep load_model for potential comparison later if needed
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import TimeDistributed  # Add TimeDistributed layer import if needed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape # Add Conv1D, MaxPooling1D, Flatten, Reshape layer imports if needed
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.metrics import matthews_corrcoef # ADDED IMPORT
import math

batch_size = 256

# --- Create model function ---
def create_binary_classification_model():
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
    return newmodel

# --- Metric Functions ---
def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def auc(y_true, y_pred):
    auc = tf.keras.metrics.AUC(name='auc') # corrected: use tf.keras.metrics.AUC
    auc.reset_state() # added to reset state at the beginning
    auc.update_state(y_true, y_pred) # update state with current batch
    return auc.result() # return result


# --- Data Loading and Preprocessing ---
Dname = '19d11014-90-labeled.csv'  # Anthrax labeled binary data file (with header)
df30 = pd.read_csv(Dname) 
dataset = df30.values
dataset = dataset.astype('float64')
timep = dataset[:, 0]
maxer = np.amax(dataset[:, 2])
print(maxer)
maxeri = maxer.astype('int')
maxchannels = maxeri
idataset = dataset[:, 2].astype(int) # Ground truth states (integer labels)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset) # Scale the entire dataset (time, current, state)
in_data = dataset_scaled[:, 1] # Use scaled current data for prediction
in_data = in_data.reshape(len(in_data), 1, 1, 1) # Reshape for TimeDistributed CNN-LSTM input


# --- Model Loading (Modified to load weights) ---
model_weights_path = 'anthrax_binary_model_weights.h5' # Path to your saved weights file
newmodel = create_binary_classification_model() # Create the model architecture
newmodel.load_weights(model_weights_path) # Load the weights into the model
print(f"Model weights loaded from: {model_weights_path}")

newmodel.summary() # Optional: Print model summary to verify architecture

# --- Prediction ---
predicted_probabilities = newmodel.predict(in_data, batch_size=batch_size, verbose=True) # Get probabilities
predicted_classes = (predicted_probabilities > 0.5).astype("int32") # Threshold at 0.5 for binary classes
predicted_binary_states = predicted_classes # predicted_classes is already binary (0 or 1)

# --- Ground Truth (for comparison and plotting) ---
target_binary_states = (idataset > 0).astype(int) # Binary ground truth states

# --- Evaluation (Optional - you can adapt/remove this for prediction script) ---
cm_dc = confusion_matrix(target_binary_states, predicted_binary_states)
print("\nConfusion Matrix - Prediction Set (Threshold 0.50):")
print(cm_dc)

print("\nPrediction Set Metrics (Threshold 0.50):")
print(classification_report(target_binary_states, predicted_binary_states))
print("MCC:", matthews_corrcoef(target_binary_states, predicted_binary_states)) # Added MCC calculation

# --- Plotting ---
lenny = 2000
ulenny = 5000
plt.figure(figsize=(30, 8)) # Increased figure height to accommodate raw current and scaled current

plt.subplot(5, 1, 1) # 5 rows for plots now
plt.plot(dataset[lenny:ulenny, 1], color='blue', label="Raw Current Data") # Raw current data
plt.title("Raw Current and Predicted/Actual States (Event-Aware Trimmed)")
plt.ylabel('Current (Raw)')
plt.legend(loc='upper right')

plt.subplot(5, 1, 2)
plt.plot(dataset_scaled[lenny:ulenny, 1], color='purple', label="Scaled Current Data (for model input)") # Scaled current data
plt.ylabel('Current (Scaled)')
plt.legend(loc='upper right')


plt.subplot(5, 1, 3)
plt.plot(target_binary_states[lenny:ulenny], color='black', label="Actual Binary States (Ground Truth)")
plt.ylabel('State (Actual)')
plt.legend(loc='upper right')

plt.subplot(5, 1, 4)
plt.plot(predicted_binary_states[lenny:ulenny], color='red', label="Predicted Binary States (Threshold 0.5)")
plt.ylabel('State (Predicted)')
plt.legend(loc='upper right')

plt.subplot(5, 1, 5) # Added subplot for probabilities
plt.plot(predicted_probabilities[lenny:ulenny], color='green', label="Predicted Probabilities (Open State)")
plt.ylabel('Probability (Open)')
plt.xlabel('Timepoint') # Moved xlabel to the last subplot
plt.legend(loc='upper right')


plt.tight_layout() # Adjust subplot parameters for a tight layout.
plt.show()

# --- Standard deviation calculation ---
x_input = dataset[:, 1]
mean_x = sum(x_input) / np.count_nonzero(x_input)

sd_x = math.sqrt(sum((x_input - mean_x)**2) / np.count_nonzero(x_input))

print("\nStandard Deviation of Raw Current Data:", sd_x)
