# Peptide classifier with features: Permutation Importance Assessment

# --- Imports ---
import numpy as np
import pickle
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, GlobalAveragePooling1D, Input, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tcn import TCN

def create_peptide_classifier_model(num_peptides, sequence_length, num_features=13, use_tcn=True, use_lstm=False):
    """
    Creates a two-branch model with either TCN, LSTM, or TCN-LSTM for one branch using state sequences 
    from translocation events as input and a second branch with translocation event kinetic features as input ultimately
    merging into an output layer for peptide classification.
    """

    # --- Input Layers ---
    sequence_input = Input(shape=(sequence_length, 1), name='sequence_input')
    feature_input = Input(shape=(num_features,), name='feature_input')

    # --- State Sequence Branch (TCN and/or LSTM) ---
    sequence_branch = sequence_input

    if use_tcn:
        sequence_branch = TCN(nb_filters=256, kernel_size=3, dilations=[1, 2, 4], return_sequences=True, dropout_rate=0.7)(sequence_branch)
        sequence_branch = BatchNormalization()(sequence_branch)

        sequence_branch = TCN(nb_filters=128, kernel_size=3, dilations=[1, 2, 4], return_sequences=True, dropout_rate=0.7)(sequence_branch)
        sequence_branch = BatchNormalization()(sequence_branch)

    if use_lstm:
        sequence_branch = LSTM(512, return_sequences=True)(sequence_branch)
        sequence_branch = BatchNormalization()(sequence_branch)
        sequence_branch = Dropout(0.7)(sequence_branch)

        sequence_branch = LSTM(256, return_sequences=True)(sequence_branch)
        sequence_branch = BatchNormalization()(sequence_branch)
        sequence_branch = Dropout(0.7)(sequence_branch)

    sequence_branch = GlobalAveragePooling1D()(sequence_branch)

    # --- Kinetic Feature Branch (Dense Layers) ---
    feature_out = Dense(32, activation='relu')(feature_input)
    feature_out = BatchNormalization()(feature_out)
    feature_out = Dropout(0.7)(feature_out)

    # --- Merge Branches ---
    merged = concatenate([sequence_branch, feature_out])

    # --- Final Dense Layers ---
    dense_out = Dense(64, activation='relu')(merged)
    dense_out = Dropout(0.7)(dense_out)
    output = Dense(num_peptides, activation='softmax')(dense_out)

    # --- Create and Compile Model ---
    model = Model(inputs=[sequence_input, feature_input], outputs=output)

    return model

def load_peptide_sequences_and_features(peptide_data_paths, peptide_labels_encoding, max_sequence_length):
    all_events_data = [] # List to hold dictionaries of events with features
    all_labels = []
    peptide_names_list = list(peptide_data_paths.keys())
    for peptide_name, filepath in peptide_data_paths.items():
        try:
            with open(filepath, 'rb') as infile:
                events_data = pickle.load(infile) # Load list of dictionaries
            labels = [peptide_labels_encoding[peptide_name]] * len(events_data)
            all_events_data.extend(events_data)
            all_labels.extend(labels)
            print(f"Loaded {len(events_data)} translocation events with features for {peptide_name} from {filepath}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading data from {filepath}: {e}")

    print(f"Total translocation events loaded (with features): {len(all_events_data)}")

    # Get state sequences from events data
    sequences = [event['states'] for event in all_events_data]

    # Get features from events data
    x_prediction_features = np.array([ # Cast directly to numpy array here
        [
            event['entropy'],
            event['first_transition_time'],
            event['avg_dwell_0'],
            event['avg_dwell_1'],
            event['var_dwell_0'],
            event['var_dwell_1'],
            event['longest_dwell_0'],
            event['longest_dwell_1'],
            event['event_duration'],
            event['probability_0'],
            event['probability_1'],
            event['ratio_0_to_1'],
            event['num_transitions']
        ]
        for event in all_events_data
    ])
    
    # Pad sequences and truncate as necessary to feed into model
    x_prediction_sequences = pad_sequences(
        [np.array(seq).reshape(-1, 1) for seq in sequences],
        maxlen=max_sequence_length,
        padding='post',
        truncating='post',
        dtype='float32',
        value=-1.0
    )

    return x_prediction_sequences, x_prediction_features, all_labels

def shuffle_column(list_of_lists, column_index):
    n = len(list_of_lists)
    if n == 0:
        return np.array()  # Return an empty NumPy array
        if column_index < 0 or column_index >= list_of_lists.shape[1]: # Use .shape for NumPy
            raise IndexError("Column index out of bounds")

    column_values = [row[column_index] for row in list_of_lists]
    random.shuffle(column_values)
    for i in range(n):
        list_of_lists[i, column_index] = column_values[i] # Use NumPy indexing

    return list_of_lists

if __name__ == "__main__":
    # --- Define some global parameters --- #
    max_sequence_length = 276 # Max sequence length from the training of the model
    num_peptides = 6 # Total number of peptides used in the training of the model
    # --- Define pickle filepaths dictionary of peptides to predict
    # Each pickle file contains list of dictionaries of segmented translocation event state sequences and 8 kinetic features
    peptide_data_paths = {
        'PeptideA': './data/peptide_A_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideB': './data/peptide_B_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideC': './data/peptide_C_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideD': './data/peptide_D_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideE': './data/peptide_E_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideF': './data/peptide_F_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl'
    }
    # Peptide encodings dictionary    
    peptide_labels_encoding = {
        'PeptideA': 0, # Numerical labels
        'PeptideB': 1,
        'PeptideC': 2,
        'PeptideD': 3,
        'PeptideE': 4,
        'PeptideF': 5
    }
    num_peptides_to_predict = len(peptide_data_paths) # Number of peptide classes
    peptides_to_predict_names_list = list(peptide_data_paths.keys()) # To access peptide names in order if neede

    # --- Model Weights Loading ---
    model_weights_path = './models/peptide_classifier_with_features_and_TCN_reduced_parameters_downsampled_peptides_A_to_F_final_weights.h5' 
    model = create_peptide_classifier_model(num_peptides, max_sequence_length, num_features=13, use_tcn=True, use_lstm=False) # Create model architecture
    model.load_weights(model_weights_path) # Load the weights into the model
    print(f"Model weights loaded from: {model_weights_path}")
    model.summary() # Optional: Print model summary to verify architecture

    # --- Load datasets and extract state sequences, features, and their corresponding labels
    x_prediction_sequences, x_prediction_features, all_labels = load_peptide_sequences_and_features(peptide_data_paths, peptide_labels_encoding, max_sequence_length)

    # --- Make Baseline prediction and compute baseline metric
    batch_size = 32
    predicted_probabilities = model.predict([x_prediction_sequences, x_prediction_features], batch_size=batch_size, verbose=True)
    predicted_labels = np.argmax(predicted_probabilities, axis=1)
    baseline_macro_f1 = f1_score(all_labels, predicted_labels, average='macro')

    # --- Shuffle and calculate feature importance --- #
    macro_f1_differences = []
    feature_names = ['entropy', 'first_transition_time', 'avg_dwell_0', 'avg_dwell_1', 'var_dwell_0', 'var_dwell_1', 'longest_dwell_0', 'longest_dwell_1', 'event_duration', 'probability_0', 'probability_1', 'ratio_0_to_1', 'num_transitions']

    for index in range(x_prediction_features.shape[1]):
        shuffled_features = shuffle_column(x_prediction_features.copy(), index)
        predicted_probabilities = model.predict([x_prediction_sequences, shuffled_features], batch_size=batch_size, verbose=True)
        predicted_labels = np.argmax(predicted_probabilities, axis=1)
        shuffled_macro_f1 = f1_score(all_labels, predicted_labels, average='macro')
        macro_f1_differences.append(baseline_macro_f1 - shuffled_macro_f1)

    # --- Print results --- #
    print("\nFeature Importance (Decrease in Macro F1-Score):")
    for feature, difference in zip(feature_names, macro_f1_differences):
        print(f"{feature}: {difference:.3f}")
        
    # --- Plot results --- #
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, macro_f1_differences)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Decrease in Macro F1-Score")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.show()
