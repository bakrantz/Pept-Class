# -*- coding: utf-8 -*-
"""
Conductance State Multi-state Classification Predictor

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
from sklearn.metrics import matthews_corrcoef # ADDED IMPORT
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score # Make sure these are imported
import math

batch_size = 256

# --- Define create_multistate_classification_model() function here ---
def create_multistate_classification_model(num_states): 
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
    return newmodel # ADDED return statement

# --- Data Loading and Preprocessing ---
Dname = '16203009-labeled-filtered-raw-data-window-size-3.csv'
df30 = pd.read_csv(Dname) # Removed header=None, assuming CSV has header now
dataset = df30.values
dataset = dataset.astype('float64')
timep = dataset[:, 0]
maxer = np.amax(dataset[:, 2])
print(maxer)
maxeri = maxer.astype('int')
num_states = maxeri + 1
idataset = dataset[:, 2].astype(int) # Ground truth states (integer labels)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset) # Scale the entire dataset (time, current, state)
in_data = dataset_scaled[:, 1] # Use scaled current data for prediction
in_data = in_data.reshape(len(in_data), 1, 1, 1) # Reshape for TimeDistributed CNN-LSTM input

# --- Model Loading (Modified to load weights) ---
model_weights_path = 'anthrax_three_state_model_weights.h5' # Path to your saved weights file
newmodel = create_multistate_classification_model(num_states) # Create the model architecture
newmodel.load_weights(model_weights_path) # Load the weights into the model
print(f"Model weights loaded from: {model_weights_path}")

newmodel.summary() # Optional: Print model summary to verify architecture

# --- Prediction ---
predicted_probabilities = newmodel.predict(in_data, batch_size=batch_size, verbose=True) # Get probabilities
predicted_classes = np.argmax(predicted_probabilities, axis=1) # Get class labels (0, 1, or 2)

# --- Ground Truth (for comparison and plotting) ---
target_multistate_states = idataset # Use original integer labels (0, 1, 2, ...) as ground truth

# --- Evaluation ---
cm_multi = confusion_matrix(target_multistate_states, predicted_classes) # Multi-class confusion matrix
print("\nMulti-Class Confusion Matrix - Prediction Set:")
print(cm_multi)

print("\nMulti-Class Classification Report - Prediction Set:")
print(classification_report(target_multistate_states, predicted_classes)) # Multi-class classification report

accuracy_multi = accuracy_score(target_multistate_states, predicted_classes) # Multi-class accuracy
print(f"Multi-Class Accuracy: {accuracy_multi:.4f}")

precision_macro_multi = precision_score(target_multistate_states, predicted_classes, average='macro') # Macro-averaged precision
recall_macro_multi = recall_score(target_multistate_states, predicted_classes, average='macro')    # Macro-averaged recall
f1_macro_multi = f1_score(target_multistate_states, predicted_classes, average='macro')        # Macro-averaged F1-score

print(f"Macro-averaged Precision: {precision_macro_multi:.4f}")
print(f"Macro-averaged Recall: {recall_macro_multi:.4f}")
print(f"Macro-averaged F1-score: {f1_macro_multi:.4f}")

# --- Plotting ---
lenny = 2000
ulenny = 10000
plt.figure(figsize=(30, 10)) # Increased height for more subplots

plt.subplot(4, 1, 1) # 4 rows for plots now
plt.plot(dataset[lenny:ulenny, 1], color='blue', label="Raw Current Data")
plt.title("Raw Current and Predicted/Actual States (Event-Aware Trimmed)")
plt.ylabel('Current (Raw)')
plt.legend(loc='upper right')

plt.subplot(4, 1, 2)
plt.plot(dataset_scaled[lenny:ulenny, 1], color='purple', label="Scaled Current Data (for model input)")
plt.ylabel('Current (Scaled)')
plt.legend(loc='upper right')

plt.subplot(4, 1, 3)
plt.plot(target_multistate_states[lenny:ulenny], color='black', label="Actual States (Ground Truth)", drawstyle='steps-mid') # Use drawstyle='steps-mid' for state plots
plt.ylabel('State (Actual)')
plt.yticks([0, 1, 2], ['Closed', 'Sub-conductance', 'Open']) # Set y-ticks and labels for 3 states
plt.legend(loc='upper right')

plt.subplot(4, 1, 4)
plt.plot(predicted_classes[lenny:ulenny], color='red', label="Predicted States", drawstyle='steps-mid') # Use drawstyle='steps-mid' for state plots
plt.ylabel('State (Predicted)')
plt.yticks([0, 1, 2], ['Closed', 'Sub-conductance', 'Open']) # Set y-ticks and labels for 3 states
plt.xlabel('Timepoint')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- (Standard deviation calculation - kept as is) ---
x_input = dataset[:, 1]
mean_x = sum(x_input) / np.count_nonzero(x_input)

sd_x = math.sqrt(sum((x_input - mean_x)**2) / np.count_nonzero(x_input))

print("\nStandard Deviation of Raw Current Data:", sd_x)
