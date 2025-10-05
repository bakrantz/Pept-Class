# Peptide classifier using translocation event conductance state sequences and features - Training
# State sequences of translocation events are fed into three layered Conv1D CNN input branch
# Features of those events are fed into a second input Dense Layer
# The two outputs of those branches are concatenated feeding into the output
# Output classification to Dense layer with softmax

# B. Krantz

# --- Import Libraries ---
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle # For loading segmented translocation events data
import numpy as np # For numerical operations
import random # For random sampling in downsampling
import hashlib
import os
import sys
import json

from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split # For train/test split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical # For one-hot encoding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # For callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Custom classes are required to access and process data in databases
# Assuming the database directory is in the parent of the directory holding the training script
common_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if common_parent_dir not in sys.path:
    sys.path.insert(0, common_parent_dir)
try:
    from database.PeptideEventsDatabase import ProcessedPeptideData, PeptideTranslocationEvents, PeptideEventsDatabase
    from database.PeptideDatabase import PeptideData, PeptideDatabase
    print("Successfully imported database classes.")
except ImportError as e:
    print(f"Error importing database classes: {e}")
    print(f"Current sys.path: {sys.path}")
    # It's better to let the calling script handle exiting if imports fail
    # For now, we'll just print and continue, but actual execution will fail if classes are truly missing.
    # raise e # Uncomment to re-raise the error and stop execution

def create_peptide_classifier_model(input_sequence_length, num_features, num_peptides):
    """
    Creates a dual-input CNN-Dense model for peptide classification.
    One input branch processes translocation event conductance state sequences with an Embedding layer
    followed by a 3-layer CNN.
    The second input branch processes flattened event-level features with a Dense layer.
    The outputs of both branches are concatenated and fed into a final Dense classifier head.

    Args:
        input_sequence_length (int): The fixed length of the input state sequences.
        num_features (int): The number of flattened event-level features for each event.
        num_peptides (int): The number of unique peptide classes for classification.

    Returns:
        tf.keras.Model: The compiled Keras functional model.
    """

    # --- Input Branch 1: Translocation Event State Sequences (CNN Branch) ---
    # Define the input layer for the state sequences. dtype='int32' is essential for Embedding.
    sequence_input = Input(shape=(input_sequence_length,), dtype='int32', name='state_sequence_input')

    # --- Embedding Layer ---
    # Map each discrete state (originally 0, 1, 2) and the padding (-1) to new integer indices.
    # After remapping: -1 -> 0 (padding), 0 -> 1, 1 -> 2, 2 -> 3.
    # So, input_dim needs to cover indices 0, 1, 2, 3. Thus, max_index (3) + 1 = 4.
    num_effective_embedding_indices = 4 # For remapped values 0 (padding), 1, 2, 3
    embedding_dim = 16 # Hyperparameter: size of the dense vector for each state. Can be tuned.

    x = layers.Embedding(input_dim=num_effective_embedding_indices,
                         output_dim=embedding_dim,
                         mask_zero=True # Crucial: tells the embedding layer to ignore index 0 (our new padding)
                        )(sequence_input)

    # NO GaussianNoise here for discrete state sequences.
    
    # First Convolutional Block
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Second Convolutional Block
    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # Third Convolutional Block
    x = layers.Conv1D(filters=256, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # GlobalMaxPooling1D to flatten the CNN output for the dense layers
    cnn_output = layers.GlobalMaxPooling1D()(x)

    # --- Input Branch 2: Event-Level Features (Dense Branch) ---
    # Define the input layer for the flattened features
    features_input = Input(shape=(num_features,), name='features_input')

    # A simple Dense layer for the features branch.
    features_dense = layers.Dense(32, activation='relu')(features_input)

    # --- Concatenate Both Branches ---
    concatenated = layers.concatenate([cnn_output, features_dense], name='concatenation_layer')

    # --- Classifier Head (Dense Layers) ---
    y = layers.Dense(64, activation='relu')(concatenated)
    y = layers.Dropout(0.45)(y)

    output_layer = layers.Dense(num_peptides, activation='softmax', name='output_layer')(y)

    # --- Create the Model ---
    model = models.Model(inputs=[sequence_input, features_input], outputs=output_layer)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model

def load_translocation_data_from_database(
    peptide_names_list: list[str],
    peptide_labels_encoding: dict,
    desired_processing_params: dict,
    raw_db_query: dict = None,
    processed_events_output_dir: str = './processed_data', # Directory to save new PKL files
    random_state: int = 42,
    downsample_to_min_events: bool = True
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]: # Added np.ndarray for features
    """
    Loads translocation event data (current sequences and event-level features)
    from the PeptideEventsDatabase, processing raw data if a suitable processed
    file doesn't exist. Labels them, and optionally downsamples to equalize
    event counts per peptide.

    Args:
        peptide_names_list (List[str]): An ordered list of peptide names (e.g., ['PeptideA', 'PeptideB']).
                                        This defines the order for labels and potential downsampling.
        peptide_labels_encoding (dict): Dictionary mapping peptide names to numerical labels (0, 1, 2...).
        desired_processing_params (dict): The exact set of processing parameters desired for the ML/DL model.
                                          If a processed PKL with these parameters exists, it's loaded.
                                          Otherwise, raw data will be processed with these parameters.
        raw_db_query (dict, optional): Query dictionary for the PeptideDatabase to filter raw data.
                                        e.g., {'noise_level': 'None'}. If None, no additional filtering.
        processed_events_output_dir (str): Directory where newly processed PKL files will be saved.
                                           Defaults to './processed_data'.
        random_state (int): Random state for reproducibility of downsampling.
        downsample_to_min_events (bool): If True, downsamples all peptides to the number of events
                                            of the peptide with the fewest events. If False, no
                                            downsampling is applied.

    Returns:
        tuple: (all_current_sequences, all_features_np, all_labels_np)
            all_current_sequences (list): List of numpy arrays, each being a translocation event current sequence.
            all_features_np (np.ndarray): NumPy array of flattened event-level features for each event.
            all_labels_np (np.ndarray): NumPy array of numerically encoded peptide labels.
    """
    # --- Calculate Absolute Paths for Databases ---
    # Get the directory of the current script (e.g., biosensor-end-to-end-current-classification)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root (e.g., /Users/bakrantz/Documents/python)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Define the path to your 'database' directory
    database_dir = os.path.join(project_root, 'database')

    # Construct the full, absolute paths to your database JSON files
    raw_db_json_path = os.path.join(database_dir, 'peptide_data.json')
    processed_events_db_json_path = os.path.join(database_dir, 'peptide_events_data.json')

    print(f"Attempting to load raw database from: {raw_db_json_path}")
    print(f"Attempting to load processed events database from: {processed_events_db_json_path}")
    # --- End Path Calculation ---

    # Initialize databases with the explicit, absolute file paths
    raw_db = PeptideDatabase(db_file=raw_db_json_path)
    processed_events_db = PeptideEventsDatabase(db_file=processed_events_db_json_path)

    # List to store tuples of (event_data_dict, event_features_array)
    all_events_and_features = []
    all_labels = []
    # List to store the extracted event-level feature names from each loaded PKL
    all_event_level_feature_names_lists = []

    print("--- Starting Data Loading/Processing from Databases ---")

    for peptide_name in peptide_names_list:
        print(f"\nSearching for data for Peptide: {peptide_name}")
        current_peptide_events_and_features = []

        # 1. Query raw database for relevant records for this peptide
        effective_raw_query = {'peptide_name': peptide_name}
        if raw_db_query:
            effective_raw_query.update(raw_db_query)

        raw_records = raw_db.retrieve_records(query=effective_raw_query)
        if not raw_records:
            print(f"No raw data records found for '{peptide_name}' with query: {effective_raw_query}. Skipping.")
            continue

        print(f"Found {len(raw_records)} raw records for '{peptide_name}'.")

        # 2. For each raw record, check for or initiate processing
        for raw_record in raw_records:
            print(f"    Handling raw record: {raw_record.data_file} (ID: {raw_record._id[:8]})")

            # Construct the processed_file name that would be generated for these parameters
            params_string = json.dumps(desired_processing_params, sort_keys=True)
            param_hash = hashlib.sha256(params_string.encode('utf-8')).hexdigest()[:16]
            sanitized_peptide_name = "".join(c if c.isalnum() else '_' for c in raw_record.peptide_name).replace('__', '_').strip('_')
            expected_processed_filename = f"{sanitized_peptide_name}_{raw_record._id[:8]}_{param_hash}.pkl"

            # Check if this specific processed file already exists in the processed events database
            existing_processed_records = processed_events_db.retrieve_processed_records(
                query={'raw_record_id': raw_record._id, 'processed_file': expected_processed_filename}
            )

            selected_processed_record = None
            if existing_processed_records:
                selected_processed_record = existing_processed_records[0]
                print(f"      Existing processed record found: {selected_processed_record.processed_file} (ID: {selected_processed_record._id[:8]})")
            else:
                print(f"      No existing processed record found with specified parameters. Initiating processing...")
                # Use the actual PeptideTranslocationEvents class
                event_processor_for_new = PeptideTranslocationEvents(raw_record, desired_processing_params)
                newly_processed_record = event_processor_for_new.process_stream(
                    output_dir=processed_events_output_dir
                )

                if newly_processed_record:
                    processed_events_db.add_processed_record(newly_processed_record)
                    selected_processed_record = newly_processed_record
                else:
                    print(f"      Failed to process raw record {raw_record._id}. Skipping.")
                    continue

            # Now, load the data from the selected (existing or newly created) PKL file
            if selected_processed_record:
                pkl_filepath = processed_events_db.get_processed_file_path(selected_processed_record._id) # Use the filepath from the record

                # Use the actual PeptideTranslocationEvents class
                event_processor_for_load = PeptideTranslocationEvents(raw_record, selected_processed_record.processing_params)
                if event_processor_for_load.load_events(pkl_filepath): # This call will populate internal data
                    # CORRECTED: Access events_data and feature_names directly
                    events_data_from_pkl = event_processor_for_load.get_events_data()
                    feature_names_from_pkl = event_processor_for_load.get_feature_names()
                    
                    if events_data_from_pkl and feature_names_from_pkl: # Check if both are not None
                        # Extract event-level feature keys in a consistent order
                        event_level_feature_keys = []
                        event_level_feature_keys.extend(feature_names_from_pkl.get('event_level_scalar', []))
                        event_level_feature_keys.extend(feature_names_from_pkl.get('event_level_vector_flat', []))
                        event_level_feature_keys.extend(feature_names_from_pkl.get('event_level_matrix_flat', []))
                        
                        # Store this list for consistency check later
                        all_event_level_feature_names_lists.append(event_level_feature_keys)

                        for event_dict in events_data_from_pkl:
                            # Extract features for the current event in the determined order
                            event_features_list = []
                            for key in event_level_feature_keys:
                                if key in event_dict:
                                    feature_value = event_dict[key]
                                    # Ensure feature_value is flattened if it's an array/list
                                    if isinstance(feature_value, (list, np.ndarray)):
                                        event_features_list.extend(np.array(feature_value).flatten().tolist())
                                    else:
                                        event_features_list.append(feature_value)
                                else:
                                    # Handle missing feature: append 0.0 or raise error
                                    # It's crucial for ML/DL that all feature vectors have the same length.
                                    print(f"Warning: Feature '{key}' not found in an event dictionary for peptide '{peptide_name}'. Appending 0.0.")
                                    event_features_list.append(0.0) # Use float for consistency

                            current_peptide_events_and_features.append((event_dict, np.array(event_features_list, dtype=np.float32)))
                        
                        if not current_peptide_events_and_features:
                            print(f"      PKL file {os.path.basename(pkl_filepath)} contained no event data after feature extraction. Skipping.")
                            continue

                        all_events_and_features.extend(current_peptide_events_and_features)
                        all_labels.extend([peptide_labels_encoding[peptide_name]] * len(current_peptide_events_and_features))
                        print(f"      Events and features accumulated for raw record {raw_record._id[:8]}: {len(current_peptide_events_and_features)}")
                    else:
                        print(f"      PKL file {os.path.basename(pkl_filepath)} missing 'events_data' or 'feature_names'. Skipping.")
                else:
                    print(f"      Failed to load events from {os.path.basename(pkl_filepath)}")
            else:
                print(f"      Error: No processed record selected for raw record ID {raw_record._id}.")

        if not current_peptide_events_and_features:
            print(f"No valid translocation events accumulated for Peptide: {peptide_name}. Skipping to next peptide.")
            continue
        
        print(f"Total events accumulated for {peptide_name}: {len(current_peptide_events_and_features)}")


    print(f"\n--- Finished Data Loading/Processing ---")
    print(f"Total translocation events loaded before downsampling: {len(all_events_and_features)}")

    if not all_events_and_features:
        print("No peptide translocation events loaded. Returning empty data.")
        return [], np.array([]), np.array([])

    # --- Feature Consistency Check ---
    if not all_event_level_feature_names_lists:
        raise ValueError("No PKL files were loaded, cannot check feature consistency.")

    # Use the first set of feature names as the reference for order and content
    master_event_level_feature_keys = all_event_level_feature_names_lists[0]
    master_feature_set = set(master_event_level_feature_keys)

    for i, feature_set_from_pkl in enumerate(all_event_level_feature_names_lists):
        current_feature_set = set(feature_set_from_pkl)
        if current_feature_set != master_feature_set:
            raise ValueError(f"Inconsistent event-level feature names detected between PKL files. "
                             f"File at index {i} has different feature names. "
                             f"Expected set: {master_feature_set}, Found set: {current_feature_set}")
        # Also check if the order is identical, which is important for consistent feature vectors
        if feature_set_from_pkl != master_event_level_feature_keys:
            # This warning is crucial if the order is not guaranteed by your processing.
            # If the processing guarantees order, this check can be removed or made an error.
            print(f"Warning: Feature names are consistent but their order differs in PKL file at index {i}. "
                  f"Ensure your feature extraction logic always uses a consistent ordering of keys. "
                  f"Using the order from the first loaded PKL file for feature vector construction.")
    
    print(f"All {len(master_event_level_feature_keys)} event-level feature names are consistent across all loaded PKL files.")


    # --- Downsampling Logic (modified to handle (event_dict, features_array) tuples) ---
    downsampled_events_and_features = []
    downsampled_labels = []

    if downsample_to_min_events:
        print("\n--- Applying Downsampling ---")
        events_and_features_by_peptide = {label: [] for label in set(all_labels)}
        for i, label in enumerate(all_labels):
            events_and_features_by_peptide[label].append(all_events_and_features[i])

        min_events = float('inf')
        min_peptide_name = "N/A"
        
        # Determine min_events based on peptides in peptide_names_list that actually have data
        actual_loaded_peptides = [p for p in peptide_names_list if peptide_labels_encoding.get(p) in events_and_features_by_peptide and len(events_and_features_by_peptide[peptide_labels_encoding[p]]) > 0]
        
        if not actual_loaded_peptides:
            print("Warning: No peptides had any loaded events. Cannot downsample. Returning all loaded data (which is empty).")
            return [], np.array([]), np.array([])

        for peptide_name_for_min in actual_loaded_peptides:
            encoded_label = peptide_labels_encoding[peptide_name_for_min]
            num_events = len(events_and_features_by_peptide[encoded_label])
            print(f"  Peptide '{peptide_name_for_min}' (Label {encoded_label}): {num_events} events")
            if num_events < min_events:
                min_events = num_events
                min_peptide_name = peptide_name_for_min
        
        if min_events == 0:
            print("  Warning: One or more peptides have 0 events after initial filtering. Cannot downsample effectively. Returning all loaded data.")
            downsampled_events_and_features = all_events_and_features
            downsampled_labels = all_labels
        else:
            print(f"  Downsampling all peptides to {min_events} events based on '{min_peptide_name}'.")
            random.seed(random_state) # Seed once for consistency across peptide sampling

            for peptide_name in peptide_names_list: # Iterate through the original ordered list
                encoded_label = peptide_labels_encoding.get(peptide_name)
                if encoded_label is not None and encoded_label in events_and_features_by_peptide:
                    peptide_events_and_features = events_and_features_by_peptide[encoded_label]
                    if len(peptide_events_and_features) > min_events:
                        sampled_tuples = random.sample(peptide_events_and_features, min_events)
                        downsampled_events_and_features.extend(sampled_tuples)
                        downsampled_labels.extend([encoded_label] * min_events)
                        print(f"    Peptide '{peptide_name}' (Label {encoded_label}) downsampled from {len(peptide_events_and_features)} to {min_events} events.")
                    else:
                        downsampled_events_and_features.extend(peptide_events_and_features)
                        downsampled_labels.extend([encoded_label] * len(peptide_events_and_features))
                        print(f"    Peptide '{peptide_name}' (Label {encoded_label}) kept {len(peptide_events_and_features)} events (no downsampling needed).")
                else:
                    print(f"    Skipping downsampling for '{peptide_name}', no data found.")
    else:
        print("\n--- Downsampling is disabled. Keeping all loaded events. ---")
        downsampled_events_and_features = all_events_and_features
        downsampled_labels = all_labels

    print(f"Total translocation events after downsampling: {len(downsampled_events_and_features)}")

    # Separate state sequences and features from the downsampled tuples
    all_state_sequences = [np.array(event_tuple[0]['states'], dtype=np.float32) for event_tuple in downsampled_events_and_features]
    all_features_np = np.array([event_tuple[1] for event_tuple in downsampled_events_and_features], dtype=np.float32)
    all_labels_np = np.array(downsampled_labels, dtype=np.int32)

    return all_state_sequences, all_features_np, all_labels_np


def plot_training_history(history, model_name, plot_filename):
    """
    Plots the training history (accuracy, loss) and saves the plot to a file.

    Args:
        history: History object returned by model.fit().
        model_name (str): Name of the model (for plot title).
        plot_filename (str): Filename to save the plot as.
    """
    # Create figure with 2 subplots (for Accuracy and Loss)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- Accuracy Plot ---
    # Ensure you use the exact key used in model.compile's metrics list (case-sensitive)
    axs[0].plot(history.history['accuracy']) # Note: Keras 3+ history.history usually uses 'accuracy' not 'Accuracy'
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # --- Loss Plot ---
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    fig.suptitle(f'Training History of {model_name}', fontsize=16) # Overall figure title
    plt.tight_layout() # Adjust layout to prevent overlap after removing a subplot
    plt.savefig(plot_filename) # Save the plot to a file
    plt.show()
    
def evaluate_model(model, x_seq_test, x_feat_test, y_test_one_hot, peptide_names_list, best_model_weights_filepath):
    """
    Evaluates the trained peptide classification model on the test set.

    Args:
        model: Trained Keras model.
        x_seq_test: Test data (padded translocation event state sequences).
        x_feat_test: Test data (scaled features).
        y_test_one_hot: One-hot encoded test labels (peptide classes).
        peptide_names_list: List of peptide names (for class label reporting).
        best_model_weights_filepath (str): Filepath to the best model weights (saved by ModelCheckpoint).
    """

    # 1. Load the best model weights (from ModelCheckpoint)
    model.load_weights(best_model_weights_filepath) # Load the best weights saved during training
    print(f"Loaded best model weights from: {best_model_weights_filepath}")

    # 2. Make predictions on the test set
    print("\n--- Making predictions on test set ---")
    y_prob_test = model.predict([x_seq_test, x_feat_test]) # Get probability predictions for test set
    y_pred_test = np.argmax(y_prob_test, axis=1) # Convert probabilities to class labels (0, 1, 2...)

    # 3. Convert one-hot encoded test labels back to class labels (0, 1, 2...)
    y_true_test = np.argmax(y_test_one_hot, axis=1) # Get true class labels from one-hot encoded labels

    # 4. Calculate and print evaluation metrics
    print("\n--- Evaluation Metrics ---")

    # Peptide Class Names for Report and Confusion Matrix labels
    target_names = peptide_names_list # Use peptide names list for class labels in report

    # Confusion Matrix
    cm_peptide = confusion_matrix(y_true_test, y_pred_test)
    print("\nPeptide Classification Confusion Matrix - Test Set:")
    print(cm_peptide)

    # Classification Report (Precision, Recall, F1-score per class)
    print("\nPeptide Classification Report - Test Set:")
    print(classification_report(y_true_test, y_pred_test, target_names=target_names, zero_division=0)) # zero_division=0 to handle cases with 0 precision/recall

    # Overall Accuracy
    accuracy_peptide = accuracy_score(y_true_test, y_pred_test)
    print(f"\nOverall Peptide Classification Accuracy: {accuracy_peptide:.4f}")

    # Macro-averaged Precision, Recall, F1-score
    precision_macro_peptide = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    recall_macro_peptide = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    f1_macro_peptide = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)

    print(f"Macro-averaged Precision: {precision_macro_peptide:.4f}")
    print(f"Macro-averaged Recall: {recall_macro_peptide:.4f}")
    print(f"Macro-averaged F1-score: {f1_macro_peptide:.4f}")

# Main Block
if __name__ == "__main__":
    # Define absolute paths for output directories relative to the script's location
    script_dir = os.path.dirname(__file__)

    processed_data_output_dir = os.path.join(script_dir, 'processed_data')
    os.makedirs(processed_data_output_dir, exist_ok=True)

    model_output_dir = os.path.join(script_dir, 'models')
    os.makedirs(model_output_dir, exist_ok=True)

    plots_output_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_output_dir, exist_ok=True)

    # --- 1. Define Peptide Data Labels, Loading, and Pre-processing Parameters ---
    # List of peptides to include in training
    peptide_names_list = [
        'guesthost_Ala',
        'guesthost_Leu',
        'guesthost_Phe',
        'guesthost_Thr',
        'guesthost_Trp',
        'guesthost_TrpDL',
        'guesthost_Tyr'
    ]
    print(f"\nPeptides to include in model: {peptide_names_list}")
    peptide_labels_encoding = {
        'guesthost_Ala': 0, # For numerical labels for training
        'guesthost_Leu': 1,
        'guesthost_Phe': 2,
        'guesthost_Thr': 3,
        'guesthost_Trp': 4,
        'guesthost_TrpDL': 5,
        'guesthost_Tyr': 6
    }

    # Processing parameters of various filterings to convert raw streams into translocation events
    desired_processing_params = {
        'high_pass_cutoff_frequency': 0, # Setting to zero disables high-pass filtering and baseline polynomial correction
        'filter_order': 3,  # This is for the existing high-pass filter
        'polynomial_degree': 2,
        'apply_polynomial_correction': True,
        'sampling_rate_hz': 400,
        'min_event_duration_ms': 5, # Events shorter than this time in ms are not included
        # Selectable low-pass filtering:
        'low_pass_filter_type': 'none',  # Can be 'none' 'median' 'bessel'
        'low_pass_filter_params': {
            'bessel': {
                'cutoff_hz': 250,  # Must be less than Nyquist 0.5*sampling_rate_hz
                'order': 4
            },
            'median': {
                'window_size': 3
            }
            # Can add parameters for other low-pass filter types here if needed
        }
    }
    print(f"\nProcessing parameters: {desired_processing_params}")

    # Query to the PeptideDatabase which holds the raw translocation event stream CSV datasets for the peptides
    raw_db_query = {
        'experimental': True,
        'nanopore_name': 'PA',
        'voltage': 70,
        'time_sampling': 400,
        'peptide_conc': {'$gte': 5, '$lte': 20}
        }

    # --- 2. Load peptide translocation event datasets from the databases, compute num_features, handle NaNs, and scale features
    random_state = 2
    print(f'\nRandom state: {random_state}')
    
    all_state_sequences, all_features_np, all_labels_np = load_translocation_data_from_database(
        peptide_names_list,
        peptide_labels_encoding,
        desired_processing_params,
        raw_db_query=raw_db_query,
        processed_events_output_dir = processed_data_output_dir, # Use the absolute path defined above
        random_state = random_state,
        downsample_to_min_events = True
    )

    # --- NEW: Remap state sequences BEFORE splitting and padding ---
    print(f"\nOriginal state sequence example (first 5 values of first sequence): {all_state_sequences[0][:5] if all_state_sequences else 'N/A'}")

    state_remapping_dict = {
        -1: 0, # Old padding (-1) becomes new padding (0)
        0: 1,  # Old state 0 becomes new state 1
        1: 2,  # Old state 1 becomes new state 2
        2: 3   # Old state 2 becomes new state 3
    }

    # Apply the remapping to all your state sequences
    # Using a list comprehension to create a new list of remapped lists
    all_state_sequences = [
        [state_remapping_dict.get(val, 0) for val in seq] # .get(val, 0) handles potential unseen values, defaults to 0 (new padding)
        for seq in all_state_sequences
    ]
    print(f"Remapped state sequence example (first 5 values of first sequence): {all_state_sequences[0][:5] if all_state_sequences else 'N/A'}")


    if all_features_np.size > 0:
        num_features = all_features_np.shape[1]
    else:
        print("Warning: No features loaded or all_features_np is empty. Setting num_features to 0.")
        num_features = 0

    # --- Handle NaNs in features BEFORE scaling (using -1.0 for imputation) ---
    if num_features > 0 and np.isnan(all_features_np).any():
        print(f"\nNaN values detected in all_features_np. Imputing with -1.0.")
        all_features_np[np.isnan(all_features_np)] = -1.0
        print("NaN imputation complete.")
        print(f"Features after imputation: Min={np.min(all_features_np):.2f}, Max={np.max(all_features_np):.2f}")
    elif num_features > 0:
        print("\nNo NaN values detected in all_features_np.")
    else:
        print("\nNo features to check for NaNs.")

    # It's crucial to scale features BEFORE splitting to ensure consistent scaling
    # across train and test sets. Fit the scaler ONLY on training data.
    if num_features > 0: # Only scale if there are features
        scaler = StandardScaler()
        # Fit on all features and transform
        all_features_scaled_np = scaler.fit_transform(all_features_np)
        print(f"\nFeatures scaled using StandardScaler. Original range: [{np.min(all_features_np):.2f}, {np.max(all_features_np):.2f}], Scaled range: [{np.min(all_features_scaled_np):.2f}, {np.max(all_features_scaled_np):.2f}] (approx. -3 to 3 for std normal)")
    else:
        all_features_scaled_np = all_features_np # If no features, keep as is
        print("\nNo features to scale.")

    # --- 3. Train/Test Split (Modified for Dual Inputs) ---
    # train_test_split can take multiple arrays as X inputs.
    # The order of outputs will match the order of inputs.
    x_seq_train, x_seq_test, \
    x_feat_train, x_feat_test, \
    y_train, y_test = train_test_split(
        all_state_sequences, # First X input (now remapped)
        all_features_scaled_np, # Second X input
        all_labels_np, # Y input
        test_size=0.2,
        random_state=random_state,
        stratify=all_labels_np
    )

    print(f"Data split into training and testing sets:")
    print(f"   Training set: {len(x_seq_train)} translocation events")
    print(f"   Testing set: {len(x_seq_test)} translocation events")

    # --- 4. Determine max length from TRAINING data only ---
    # Using list comprehension for robustness
    max_train_sequence_length = max(len(seq) for seq in x_seq_train)
    print(f"Max sequence length in training data (for padding): {max_train_sequence_length}")
    max_effective_sequence_length = min(max_train_sequence_length, 1300) # Truncate longer events to keep efficient

    # --- 5. Pad Sequences ---
    # Pad both training and test sets to max_train_sequence_length
    # Crucially: dtype='int32' and value=0 for the new padding scheme
    x_seq_train_padded = pad_sequences(x_seq_train, maxlen=max_effective_sequence_length, padding='post', dtype='int32', value=0)
    x_seq_test_padded = pad_sequences(x_seq_test, maxlen=max_effective_sequence_length, padding='post', dtype='int32', value=0)

    # --- REMOVE: No longer need to expand_dims here for Embedding layer ---
    # The Embedding layer expects (batch_size, sequence_length)
    # and outputs (batch_size, sequence_length, embedding_dim), which is 3D already for Conv1D.
    # x_seq_train_padded = np.expand_dims(x_seq_train_padded, axis=-1)
    # x_seq_test_padded = np.expand_dims(x_seq_test_padded, axis=-1)

    print(f"x_seq_train_padded shape: {x_seq_train_padded.shape}")
    print(f"x_seq_test_padded shape: {x_seq_test_padded.shape}")

    # --- 6. One-Hot Encode Labels ---
    # Use the number of unique labels from your encoding dictionary
    num_peptides = len(peptide_labels_encoding)
    y_train_one_hot = to_categorical(y_train, num_classes=num_peptides)
    y_test_one_hot = to_categorical(y_test, num_classes=num_peptides)

    print(f"train_labels_one_hot shape: {y_train_one_hot.shape}")
    print(f"test_labels_one_hot shape: {y_test_one_hot.shape}")

    # Check for NaNs (good practice, though less likely with current sequences after scaling)
    print(f"NaNs in x_seq_train_padded: {np.isnan(x_seq_train_padded).any()}") # This check should now be false if padding works
    print(f"NaNs in x_seq_test_padded: {np.isnan(x_seq_test_padded).any()}")

    # --- 7. Training Callbacks ---
    model_name = "peptide_classifier_CNN_Dense_state_sequences_with_features_guesthost_20ms" # Descriptive model name

    # Use the absolute model_output_dir for file paths
    best_model_weights_filepath = os.path.join(model_output_dir, f'{model_name}_best_weights.weights.h5')
    final_model_weights_filepath = os.path.join(model_output_dir, f'{model_name}_final_weights.weights.h5')

    # Use the absolute plots_output_dir for plot filepath
    plot_filepath = os.path.join(plots_output_dir, f'{model_name}_training_history.png')

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20, # Increased patience as deep models can take longer to converge
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        filepath=best_model_weights_filepath, # Save best weights to best_weights filepath
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min' # Specify mode for monitor='val_loss'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by a factor of 0.5
        patience=5,  # Reduce LR if val_loss doesn't improve for 5 epochs
        min_lr=0.000001, # Minimum learning rate
        verbose=1,
        mode='min'
    )

    callbacks_list = [model_checkpoint, early_stopping, reduce_lr]

    # --- 8. Create Compiled Model ---
    model = create_peptide_classifier_model(max_effective_sequence_length, num_features, num_peptides)
    model.summary()

    # --- 9. Train Model ---
    epochs = 100
    batch_size = 32

    print("\n--- Starting Model Training ---")

    history = model.fit(
        # Provide a list of inputs, matching the order of Input layers in your model
        [x_seq_train_padded, x_feat_train], # Training data: [state sequences, features]
        y_train_one_hot,                    # Training labels (one-hot encoded peptide classes)
        epochs=epochs,
        batch_size=batch_size,
        # Provide validation_data as a tuple: ([list of validation inputs], validation labels)
        validation_data=([x_seq_test_padded, x_feat_test], y_test_one_hot),
        callbacks=callbacks_list,
        verbose=1 # Set to 1 to see training progress
    )

    print("\n--- End Model Training ---")

    # --- 10. Evaluate Model on Test Set ---
    print("\n--- Evaluating Model on Test Set ---")
    evaluate_model(model, x_seq_test_padded, x_feat_test, y_test_one_hot, peptide_names_list, best_model_weights_filepath)

    # --- 11. Save Model Weights and Plot Training History ---
    model.save_weights(final_model_weights_filepath)
    print(f"Trained model weights (final epoch) saved to: {final_model_weights_filepath}")
    plot_training_history(history, model_name=model_name, plot_filename=plot_filepath)

    print("\nPeptide classifier training script completed.")
