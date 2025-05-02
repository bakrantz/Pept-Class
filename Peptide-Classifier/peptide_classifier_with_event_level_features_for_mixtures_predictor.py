# Peptide classifier predictor with event-level features for evaluating sample mixtures

# --- Imports ---
import numpy as np
import pickle
import tensorflow as tf
import random
import math
import matplotlib.pyplot as plt

from scipy.stats import entropy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, GlobalAveragePooling1D, Input, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tcn import TCN
from collections import Counter

def create_peptide_classifier_model(num_peptides, sequence_length, num_features=12, use_tcn=True, use_lstm=False):
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
        sequence_branch = TCN(nb_filters=256, kernel_size=3, dilations=[1, 2, 4], return_sequences=True, dropout_rate=0.3)(sequence_branch) # Assuming 0.3 dropout was the best
        sequence_branch = BatchNormalization()(sequence_branch)

        sequence_branch = TCN(nb_filters=128, kernel_size=3, dilations=[1, 2, 4], return_sequences=True, dropout_rate=0.3)(sequence_branch) # Assuming 0.3 dropout was the best
        sequence_branch = BatchNormalization()(sequence_branch)

    if use_lstm:
        sequence_branch = LSTM(512, return_sequences=True, dropout=0.3)(sequence_branch)
        sequence_branch = BatchNormalization()(sequence_branch)

        sequence_branch = LSTM(256, return_sequences=True, dropout=0.3)(sequence_branch)
        sequence_branch = BatchNormalization()(sequence_branch)


    sequence_branch = GlobalAveragePooling1D()(sequence_branch)

    # --- Kinetic Feature Branch (Dense Layers) ---
    feature_out = Dense(32, activation='relu')(feature_input)
    feature_out = BatchNormalization()(feature_out)
    feature_out = Dropout(0.3)(feature_out) # Assuming 0.3 dropout was the best

    # --- Merge Branches ---
    merged = concatenate([sequence_branch, feature_out])

    # --- Final Dense Layers ---
    dense_out = Dense(64, activation='relu')(merged)
    dense_out = Dropout(0.3)(dense_out) # Assuming 0.3 dropout was the best
    output = Dense(num_peptides, activation='softmax')(dense_out)

    # --- Create and Compile Model ---
    model = Model(inputs=[sequence_input, feature_input], outputs=output)

    return model

def create_mixed_sample(peptide_data_paths, peptide_labels_encoding, sample_fractions, total_events_in_sample, max_sequence_length=None):
    mixed_sample_events = []
    mixed_sample_labels = []
    for peptide_name, fraction in sample_fractions.items():
        peptide_filepath = peptide_data_paths[peptide_name]
        with open(peptide_filepath, 'rb') as infile:
            events_data = pickle.load(infile)
        num_events_to_select = int(total_events_in_sample * fraction)
        selected_events = random.sample(events_data, num_events_to_select)
        mixed_sample_events.extend(selected_events)
        mixed_sample_labels.extend([(peptide_labels_encoding[peptide_name], fraction)] * num_events_to_select)

    # Extract state sequences and features
    state_sequences = [event['states'] for event in mixed_sample_events]
    features = np.array([
        [
            # event['entropy'], # Not an impactful feature
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
        for event in mixed_sample_events
    ])

    # Pad sequences if max_sequence_length is provided
    if max_sequence_length is not None:
        padded_state_sequences = pad_sequences(
            [np.array(seq).reshape(-1, 1) for seq in state_sequences],
            maxlen=max_sequence_length,
            padding='post',
            truncating='post',
            dtype='float32',
            value=-1.0
        )
        return padded_state_sequences, features, mixed_sample_labels
    else:
        return state_sequences, features, mixed_sample_labels

def classify_with_confidence_threshold(predicted_probabilities, confidence_threshold, class_names=None):
    """
    Determines the predicted classification for each event based on predicted probabilities
    and a threshold.

    Args:
        predicted_probabilities (np.array): A 2D numpy array where each row represents
            an event and each column represents the probability of that event
            belonging to a specific class.
        threshold (float): The minimum probability for the prediction to be considered valid.
            If the maximum probability for an event is below this threshold, the
            prediction will be 'None'.
        class_names (list, optional): A list of class names corresponding to the
            columns of the predicted_probabilities array. If provided, the function
            will return class names instead of class indices. Defaults to None.

    Returns:
        np.array: A 1D numpy array of predicted classifications (either class names or
            indices, or 'None' if the maximum probability is below the threshold).
    """
    max_probabilities = np.max(predicted_probabilities, axis=1)
    predicted_classes_indices = np.argmax(predicted_probabilities, axis=1)
    num_events = predicted_probabilities.shape[0]
    predicted_classifications = np.empty(num_events, dtype=object)  # Use object dtype to allow 'None'

    for i in range(num_events):
        if max_probabilities[i] >= confidence_threshold:
            if class_names is not None:
                predicted_classifications[i] = class_names[predicted_classes_indices[i]]
            else:
                predicted_classifications[i] = predicted_classes_indices[i]
        else:
            predicted_classifications[i] = 'None'

    return predicted_classifications


def evaluate_vote_pattern(predictions, true_label):
    """Evaluates the vote pattern for a test peptide."""

    unique_labels, counts = np.unique(predictions, return_counts=True)
    vote_counts = dict(zip(unique_labels, counts))

    # Top-1 accuracy
    winning_label = max(vote_counts, key=vote_counts.get)
    top1_accuracy = 1 if winning_label == true_label else 0

    # Confidence score
    confidence = vote_counts.get(winning_label, 0) / len(predictions)

    # Entropy
    probabilities = counts / len(predictions)
    vote_entropy = entropy(probabilities)

    # Rank of true label
    ranked_labels = sorted(vote_counts, key=vote_counts.get, reverse=True)
    rank = ranked_labels.index(true_label) + 1

    return top1_accuracy, confidence, vote_entropy, rank

# --- Main Script ---

if __name__ == "__main__":
    # --- Define some global parameters --- #
    max_sequence_length = 1549 # Max sequence length from the training of the model
    num_peptides = 7 # Total number of peptides used in the training of the model
    num_features = 12 # Number of event-level features used

    # --- Define pickle filepaths dictionary of peptides to predict
    # Each pickle file contains list of dictionaries of segmented translocation event state sequences and 13 kinetic features
    peptide_data_paths = {
        'PeptideA': './data/peptide_A_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideB': './data/peptide_B_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideC': './data/peptide_C_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideD': './data/peptide_D_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideE': './data/peptide_E_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideF': './data/peptide_F_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
        'PeptideG': './data/peptide_G_simulated_single_channel_data_for_prediction_150s_length_5_filtered_with_13_features.pkl'
    }
    # Peptide encodings dictionary
    peptide_labels_encoding = {
        'PeptideA': 0, # Numerical labels
        'PeptideB': 1,
        'PeptideC': 2,
        'PeptideD': 3,
        'PeptideE': 4,
        'PeptideF': 5,
        'PeptideG': 6
    }
    peptide_labels_encoding_reversed = {v: k for k, v in peptide_labels_encoding.items()}
    peptides_to_predict_names_list = list(peptide_data_paths.keys()) # To access peptide names in order if needed

    # --- Define the mixed sample parameters ---
    sample_fractions = {'PeptideA': 0.6, 'PeptideD': 0.4} # The peptide mixture specified as fractions
    total_events_in_sample = 500

    # --- Load the trained StandardScaler ---
    scaler_filepath = './models/peptide_classifier_with_12_event_level_features_and_TCN_peptides_A_to_G_feature_scaler.pkl' # Path to saved scaler
    try:
        with open(scaler_filepath, 'rb') as f:
            scaler = pickle.load(f)
        print(f"StandardScaler loaded from: {scaler_filepath}")
    except FileNotFoundError:
        print(f"Error: StandardScaler file not found at {scaler_filepath}. Make sure to train the model and save the scaler first.")
        exit() # Exit if the scaler is not found

    # --- Model Weights Loading ---
    model_weights_path = './models/peptide_classifier_with_12_event_level_features_and_TCN_peptides_A_to_G_best_weights.h5'
    # Make sure the model architecture matches the trained model, especially dropout rates
    model = create_peptide_classifier_model(num_peptides, max_sequence_length, num_features=12, use_tcn=True, use_lstm=False)
    model.load_weights(model_weights_path) # Load the weights into the model
    print(f"Model weights loaded from: {model_weights_path}")
    # model.summary() # Optional: Print model summary to verify architecture - uncomment if needed

    # --- Generate the mixed sample ---
    mixed_sequences, mixed_features, mixed_sample_labels_with_fractions = create_mixed_sample(
        peptide_data_paths,
        peptide_labels_encoding,
        sample_fractions,
        total_events_in_sample,
        max_sequence_length
    )
    print(f"Shape of raw mixed_sequences: {mixed_sequences.shape}")
    print(f"Shape of raw mixed_features: {mixed_features.shape}")

    # --- Standardize the mixed features using the loaded scaler ---
    mixed_features_scaled = scaler.transform(mixed_features)
    print(f"Shape of scaled mixed_features: {mixed_features_scaled.shape}")

    # --- Separate numerical labels and fractions ---
    true_numerical_labels = np.array([label for label, fraction in mixed_sample_labels_with_fractions])
    # The 'true_fractions' from the mixed_sample_labels_with_fractions is per-event,
    # which is not what we need for aggregate fraction evaluation.
    # We calculate the actual fraction of each peptide in the mixed sample below.

    print("Shape of true_numerical_labels:", true_numerical_labels.shape)


    # --- Perform event-level prediction with confidence threshold ---
    confidence_threshold = 0.20
    # Use the SCALED features for prediction
    predicted_probabilities = model.predict([mixed_sequences, mixed_features_scaled], batch_size=32, verbose=True)
    predicted_classes_with_none = classify_with_confidence_threshold(
        predicted_probabilities, confidence_threshold, class_names=None
    )
    confident_indices = np.where(predicted_classes_with_none != 'None')[0]
    confident_predicted_labels_numerical = np.array([
        predicted_classes_with_none[i] for i in confident_indices if predicted_classes_with_none[i] != 'None'
    ], dtype=int)
    confident_true_labels = true_numerical_labels[confident_indices] # Get the true labels corresponding to confident predictions

    print("Shape of predicted_probabilities:", predicted_probabilities.shape)
    print("Shape of predicted_classes with 'None':", predicted_classes_with_none.shape)
    print("Number of confident predictions:", len(confident_predicted_labels_numerical))

    # --- Evaluation on Confident Predictions ---
    predicted_class_counts = Counter(confident_predicted_labels_numerical)
    total_confident_predictions = len(confident_predicted_labels_numerical)

    class_fractions = {}
    if total_confident_predictions > 0:
        class_fractions = {
            peptide_labels_encoding_reversed[label]: count / total_confident_predictions
            for label, count in predicted_class_counts.items()
        }

    print("Counts of each confidently predicted class:", predicted_class_counts)
    print("Fraction of total confident predictions for each class:", class_fractions)

    # Get the actual fractions per true label in the GENERATED mixed sample
    actual_fractions_by_label = {}
    unique_true_labels = np.unique(true_numerical_labels)
    for label in unique_true_labels:
        # Count occurrences of each true label in the mixed sample
        actual_fractions_by_label[label] = np.sum(true_numerical_labels == label) / len(true_numerical_labels)

    print("\nActual fractions of each peptide in the mixture (based on generated sample):")
    print({peptide_labels_encoding_reversed[k]: v for k, v in actual_fractions_by_label.items()}) # Print with names


    # Calculate MAE, MSE, RMSE based on all truly present peptides
    mae_total = 0
    mse_total = 0
    # We should compare against all possible labels that were *actually* in the mixture,
    # even if some weren't confidently predicted.
    actual_peptide_labels_in_mixture_numerical = list(actual_fractions_by_label.keys())


    for label_num in actual_peptide_labels_in_mixture_numerical:
        peptide_name = peptide_labels_encoding_reversed[label_num]
        actual_fraction = actual_fractions_by_label[label_num]
        predicted_fraction = class_fractions.get(peptide_name, 0.0)  # Default to 0 if not confidently predicted
        mae_total += abs(predicted_fraction - actual_fraction)
        mse_total += (predicted_fraction - actual_fraction) ** 2

    if actual_peptide_labels_in_mixture_numerical: # Check if there were any actual labels
        mae = mae_total / len(actual_peptide_labels_in_mixture_numerical)
        mse = mse_total / len(actual_peptide_labels_in_mixture_numerical)
        rmse = math.sqrt(mse)
        print(f"\nMean Absolute Error (MAE) of predicted fractions (confident): {mae:.4f}")
        print(f"Mean Squared Error (MSE) of predicted fractions (confident): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE) of predicted fractions (confident): {rmse:.4f}")
    else:
        print("\nNo actual labels in the mixture to compare against for fraction metrics.")


    # Per-Event Classification Report (Confident Predictions Only)
    if confident_true_labels.size > 0 and confident_predicted_labels_numerical.size > 0:
        # Define all possible numerical labels from the training set
        all_possible_labels_numerical = sorted(peptide_labels_encoding.values())
        # Generate target names based on all possible peptide labels
        target_names = [peptide_labels_encoding_reversed[label] for label in all_possible_labels_numerical]

        print("\nPer-Event Classification Report (Confident Predictions Only):")
        # Use all_possible_labels_numerical for labels in report and confusion matrix
        print(classification_report(confident_true_labels, confident_predicted_labels_numerical,
                                     target_names=target_names, zero_division=0,
                                     labels=all_possible_labels_numerical))

        print("\nConfusion Matrix (Confident Predictions Only):")
        print(confusion_matrix(confident_true_labels, confident_predicted_labels_numerical,
                               labels=all_possible_labels_numerical))
    else:
        print("\nNo confident predictions made above the threshold. Classification Report and Confusion Matrix not generated.")

    # --- Plotting Predicted vs. Actual Fractions ---
    # Get all unique peptide labels from the defined sample fractions and confident predictions
    actual_peptide_labels_defined = set(sample_fractions.keys()) # Labels defined in the mixture
    predicted_peptide_labels_confident = set(class_fractions.keys()) # Labels confidently predicted

    # Get all unique peptide names that should be in the plot
    all_peptide_labels_for_plot = sorted(list(actual_peptide_labels_defined.union(predicted_peptide_labels_confident)))

    # Get actual fractions for all relevant peptides (based on the initial sample_fractions definition)
    actual_fractions_for_plot = np.array([sample_fractions.get(label_name, 0)
                                          for label_name in all_peptide_labels_for_plot])

    # Get predicted fractions for all relevant peptides (based on confident predictions)
    predicted_fractions_for_plot = np.array([class_fractions.get(label_name, 0)
                                           for label_name in all_peptide_labels_for_plot])

    x = np.arange(len(all_peptide_labels_for_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7)) # Adjust figure size
    rects1 = ax.bar(x - width/2, actual_fractions_for_plot, width, label='Actual Fraction (Defined Mixture)')
    rects2 = ax.bar(x + width/2, predicted_fractions_for_plot, width, label='Predicted Fraction (Confident Predictions)')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Fraction')
    ax.set_xlabel('Peptide')
    ax.set_title('Comparison of Actual and Predicted Peptide Fractions in Mixture')
    ax.set_xticks(x)
    ax.set_xticklabels(all_peptide_labels_for_plot, rotation=45, ha="right") # Rotate labels
    ax.legend()

    # Add labels to the bars - ensure you handle cases where rects1 or rects2 might be empty
    if rects1:
        ax.bar_label(rects1, fmt='%.2f', padding=3)
    if rects2:
        ax.bar_label(rects2, fmt='%.2f', padding=3)


    fig.tight_layout()
    plt.show()
