# Peptide classifier with features predictor
# Uses vote pattern aggregation to predict translocation event streams
# --- Imports ---
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import entropy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, GlobalAveragePooling1D, Input, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tcn import TCN

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

def load_prediction_peptide_sequences_and_features(peptide_filepath, peptide_name, max_sequence_length, peptide_labels_encoding):
    events_data = [] # List to hold dictionaries of events with features
    labels = []
    try:
        with open(peptide_filepath, 'rb') as infile:
            events_data = pickle.load(infile)
            # Ensure peptide_name exists in the encoding dictionary
            if peptide_name not in peptide_labels_encoding:
                 raise ValueError(f"Peptide name '{peptide_name}' not found in peptide_labels_encoding.")
            peptide_label = peptide_labels_encoding[peptide_name]
            labels = [peptide_label] * len(events_data)
            print(f"Loaded {len(events_data)} translocation events with features for {peptide_name} from {peptide_filepath}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found: {peptide_filepath}")
    except Exception as e:
        raise Exception(f"Error loading data from {peptide_filepath}: {e}")

    # Get sequences from events data
    sequences = [event['states'] for event in events_data]

    # Get features from events data
    x_prediction_features = np.array([ # Cast directly to numpy array here
        [
            # event['entropy'],
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
        for event in events_data
    ])

    # Pad sequences
    x_prediction_sequences = pad_sequences(
        [np.array(seq).reshape(-1, 1) for seq in sequences],
        maxlen=max_sequence_length,
        padding='post',
        truncating='post',
        dtype='float32',
        value=-1.0
    )

    return x_prediction_sequences, x_prediction_features, labels

# Modified evaluate_vote_pattern to correctly handle rank when true label has no predictions
def evaluate_vote_pattern(predictions, true_label, num_possible_labels):
    """
    Evaluates the vote pattern for a test peptide stream.

    Args:
        predictions (np.array): A 1D numpy array of predicted class labels for events in the stream.
        true_label (int): The true class label for the stream.
        num_possible_labels (int): The total number of possible peptide classes.

    Returns:
        tuple: (top1_accuracy, confidence, vote_entropy, rank)
    """

    # Count votes for all possible labels (0 to num_possible_labels - 1)
    vote_counts_all_labels = np.bincount(predictions, minlength=num_possible_labels)

    # Top-1 accuracy
    # Find the label with the maximum vote count among *all* possible labels
    winning_label = np.argmax(vote_counts_all_labels)
    top1_accuracy = 1 if winning_label == true_label else 0

    # Confidence score - based on the winning label's count
    total_votes = len(predictions)
    if total_votes > 0:
        confidence = vote_counts_all_labels[winning_label] / total_votes
    else:
        confidence = 0 # Handle case with no predictions

    # Entropy (on the distribution of votes across *all* possible labels)
    if total_votes > 0:
         probabilities = vote_counts_all_labels / total_votes
         # Use base 2 for entropy if desired, or default is natural log
         vote_entropy = entropy(probabilities, base=2) # Often base 2 for classification
    else:
        vote_entropy = 0 # No votes, entropy is 0

    # Rank of true label
    # Get the vote count for the true label
    true_label_count = vote_counts_all_labels[true_label]

    # Count how many labels have a strictly higher vote count than the true label
    # Add 1 to this count to get the rank.
    # If there are ties, this calculation places the true label among others with the same count.
    rank = 1 # Rank starts at 1
    for label, count in enumerate(vote_counts_all_labels): # Iterate through all possible labels and their counts
        if count > true_label_count:
            rank += 1
        # Optional: For tied counts, you could add secondary sorting by label number
        # elif count == true_label_count and label < true_label:
        #     rank += 1 # If tied count, rank labels with smaller number higher

    return top1_accuracy, confidence, vote_entropy, rank


# --- Main Script ---

if __name__ == "__main__":
    # --- Define some global parameters --- #
    max_sequence_length = 1549 # Max sequence length from the training of the model
    num_peptides = 7 # Total number of peptides used in the training of the model

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
    num_peptides_to_predict = len(peptide_data_paths) # Number of peptide classes being predicted (should match training)
    peptides_to_predict_names_list = list(peptide_data_paths.keys()) # To access peptide names in order if neede

    # --- Model Weights Loading ---
    model_weights_path = './models/peptide_classifier_with_12_event_level_features_and_TCN_peptides_A_to_G_best_weights.h5'
    model = create_peptide_classifier_model(num_peptides, max_sequence_length, num_features=12, use_tcn=True, use_lstm=False) # Create model architecture
    model.load_weights(model_weights_path) # Load the weights into the model
    print(f"Model weights loaded from: {model_weights_path}")
    # model.summary() # Optional: Print model summary to verify architecture

    # --- Load trained feature Standard Scaler ---
    scaler_filepath = './models/peptide_classifier_with_12_event_level_features_and_TCN_peptides_A_to_G_feature_scaler.pkl' # Path to saved scaler
    try:
        with open(scaler_filepath, 'rb') as f:
            scaler = pickle.load(f)
        print(f"StandardScaler loaded from: {scaler_filepath}")
    except FileNotFoundError:
        print(f"Error: StandardScaler file not found at {scaler_filepath}. Make sure to train the model and save the scaler first.")
        exit() # Exit if the scaler is not found

    overall_predictions = []
    all_top1_accuracies = []
    all_confidences = []
    all_entropies = []
    all_ranks = []

    # Loop through each peptide file to predict the pure stream
    for peptide_key in peptide_data_paths:
        peptide_filepath = peptide_data_paths[peptide_key]
        # Pass peptide_labels_encoding to load_prediction_peptide_sequences_and_features
        x_prediction_sequences, x_prediction_features, labels = load_prediction_peptide_sequences_and_features(
            peptide_filepath, peptide_key, max_sequence_length, peptide_labels_encoding # Pass encoding here
            )

        # Standardize the features using the loaded scaler
        x_prediction_features_scaled = scaler.transform(x_prediction_features)

        batch_size = 32
        # Use the SCALED features for prediction
        predicted_probabilities = model.predict([x_prediction_sequences, x_prediction_features_scaled], batch_size=batch_size, verbose=True)
        predicted_classes = np.argmax(predicted_probabilities, axis=1)

        # Voting Logic
        # Note: Using np.bincount is efficient for counting occurrences of non-negative integers
        # It will count votes for all labels from 0 up to max(predicted_classes) or num_peptides_to_predict if minlength is set.
        vote_counts_array = np.bincount(predicted_classes, minlength=num_peptides) # Use num_peptides for minlength

        # predicted_peptide_label is the numerical label with the most votes
        predicted_peptide_label = np.argmax(vote_counts_array)

        # Convert numerical labels back to peptide names for reporting
        reversed_peptide_labels_encoding = {value: key for key, value in peptide_labels_encoding.items()}
        predicted_peptide_name = reversed_peptide_labels_encoding[predicted_peptide_label]
        # labels[0] is the true numerical label for this pure stream
        actual_peptide_name = reversed_peptide_labels_encoding[labels[0]]

        prediction_instance = {'actual': actual_peptide_name, 'predicted': predicted_peptide_name}
        overall_predictions.append(prediction_instance)
        print(f"\nOverall stream prediction for {actual_peptide_name}:\nPredicted as {predicted_peptide_name}.")


        # Calculate Vote Pattern Metrics using the modified function
        # Pass the total number of possible peptides
        top1_accuracy, confidence, vote_entropy, rank = evaluate_vote_pattern(
            predicted_classes, labels[0], num_peptides # Pass num_peptides
            )

        all_top1_accuracies.append(top1_accuracy)
        all_confidences.append(confidence)
        all_entropies.append(vote_entropy)
        all_ranks.append(rank)

        print(f"Top1 accuracy for {peptide_key}: {top1_accuracy}")
        print(f"Confidence for {peptide_key}: {confidence:.4f}")
        print(f"Vote entropy for {peptide_key}: {vote_entropy:.4f}")
        print(f"Rank for {peptide_key}: {rank}")

        # Optional: Print detailed classification report and confusion matrix for event-level predictions
        # This shows event-by-event performance within the stream
        # print(f"\nEvent-level Evaluation for {peptide_key}:")
        # print(classification_report(labels, predicted_classes, zero_division=0,
        #                            target_names=peptides_to_predict_names_list,
        #                            labels=list(peptide_labels_encoding.values())))
        # print("Event-level Confusion Matrix:\n", confusion_matrix(labels, predicted_classes,
        #                                                          labels=list(peptide_labels_encoding.values())))

        # Plotting Vote Counts
        plt.figure()
        # Ensure all possible peptide classes are represented on the x-axis
        peptide_indices = list(peptide_labels_encoding.values())
        peptide_names = list(peptide_labels_encoding.keys())

        # Ensure vote_counts_array has counts for all possible peptides
        # np.bincount with minlength already does this based on num_peptides
        plt.bar(peptide_indices, vote_counts_array)
        plt.xticks(peptide_indices, peptide_names, rotation=45, ha="right")
        plt.xlabel("Peptide Class")
        plt.ylabel("Vote Counts")
        plt.title(f"Vote Counts for Pure Stream of {actual_peptide_name}")
        plt.tight_layout()
        plt.show()

    # Print formatted readable table of vote pattern metrics per peptide
    print("\n--- Vote Pattern Metrics per Peptide Stream ---")
    print("Peptide  | Top1 Accuracy | Confidence | Vote Entropy | Rank")
    print("---------|---------------|------------|--------------|------")
    # Ensure the order matches the peptide_data_paths dictionary keys
    # Loop through peptides in the order they were processed to match the lists of results
    for index, peptide_key in enumerate(peptide_data_paths.keys()):
         top1_accuracy = all_top1_accuracies[index]
         confidence = all_confidences[index]
         vote_entropy = all_entropies[index]
         rank = all_ranks[index]
         print(f"{peptide_key:<9}| {top1_accuracy:<13} | {confidence:<10.4f} | {vote_entropy:<12.4f} | {rank:<5}")


    # Print overall vote pattern metrics
    print("\n--- Overall Vote Pattern Metrics (Averaged Across Peptides) ---")
    print(f"Average Top-1 Accuracy: {np.mean(all_top1_accuracies):.4f}")
    print(f"Average Confidence: {np.mean(all_confidences):.4f}")
    print(f"Average Entropy: {np.mean(all_entropies):.4f}")
    print(f"Average Rank: {np.mean(all_ranks):.4f}")
