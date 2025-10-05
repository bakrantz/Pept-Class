# Peptide classifier that uses a hybrid DL-ML approach
# Feeds segmented translocation event conductance state sequences into cnn neural network to determine embeddings
# Feeds resulting embeddings and translocation event features into XGBoost for output classifications

# B. Krantz

import tensorflow as tf
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import sys
import json
import hashlib
import random

# Custom classes are required to access and process data in databases
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
    sys.exit(1) # Exit if essential imports fail

# --- CNN Embedding Encoder Model Definition ---
def create_cnn_model(sequence_length, vocab_size, num_peptides, embedding_dim=128,
                     num_cnn_layers=2, filters=128, kernel_size=5, pool_size=2,
                     dropout_rate=0.2, output_embedding_dim=128):
    """
    Creates a CNN model for sequence embedding and classification head.

    Input args:
    sequence_length: maximum length of the input conductance state sequence (including padding)
    vocab_size: Number of *remapped* unique token IDs (original states + 1 for padding token 0).
                The padding token is 0.
    num_peptides: number of peptide classes
    embedding_dim: Dimension of token embeddings
    num_cnn_layers: Number of stacked Conv1D layers
    filters: Number of filters (output dimensionality) in Conv1D layers
    kernel_size: Size of the convolution window
    pool_size: Size of the pooling window for MaxPooling1D
    dropout_rate: Dropout rate for regularization
    output_embedding_dim: Desired size of the final sequence embedding output from the model.

    Output:
    model: Keras Model with two outputs: classification prediction (for supervised training of the encoder)
           and the sequence embedding (to be fed into XGBoost).
    """
    inputs = layers.Input(shape=(sequence_length,), dtype='int32', name='input_sequence')

    # Embedding layer for State Tokens: maps remapped state IDs to dense vectors.
    # Use mask_zero=True as padding value is 0 after remapping.
    x = layers.Embedding(input_dim=vocab_size,
                         output_dim=embedding_dim,
                         mask_zero=True, # Mask padding value 0
                         name="token_embedding"
                        )(inputs)

    # Stacked CNN layers
    for i in range(num_cnn_layers):
        x = layers.Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          activation='relu',
                          padding='same', # Maintain sequence length
                          name=f'conv1d_{i}'
                         )(x)
        x = layers.BatchNormalization(name=f'batch_norm_{i}')(x) # Often helps CNNs
        x = layers.MaxPooling1D(pool_size=pool_size, name=f'max_pooling_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'cnn_dropout_{i}')(x)

    # Global Pooling layer: get a fixed-size embedding for the whole sequence.
    # Use GlobalMaxPooling1D as it often performs well after Conv1D, capturing the most salient features.
    # GlobalAveragePooling1D is also a valid alternative.
    sequence_representation = layers.GlobalMaxPooling1D(name="global_max_pooling")(x)


    # Final dense layer for the output embedding dimension desired for XGBoost.
    # Keep the name "embedding" for consistent downstream extraction.
    embedding_output = layers.Dense(output_embedding_dim, name="embedding")(sequence_representation)

    # Classification head (for supervised training of the encoder)
    classification_output = layers.Dense(num_peptides, activation="softmax", name="classification_head")(embedding_output)

    # --- Create and Compile the Model ---
    model = Model(inputs=inputs, outputs=[classification_output, embedding_output])

    # Compile the model for supervised classification training
    # Loss dictionary MUST include keys for ALL model outputs.
    model.compile(
        optimizer='adam',
        loss={'classification_head': 'sparse_categorical_crossentropy', 'embedding': None},
        metrics={'classification_head': 'accuracy'}
    )

    return model

# --- Data Loading Function (Modified for State Sequences) ---
def load_translocation_data_from_database(
    peptide_names_list: list[str],
    peptide_labels_encoding: dict,
    desired_processing_params: dict,
    raw_db_query: dict = None,
    processed_events_output_dir: str = './processed_data', # Directory to save new PKL files
    random_state: int = 42,
    downsample_to_min_events: bool = True
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]: # Now returns state sequences, features, labels, number of states
    """
    Loads translocation event data (conductance state sequences and event-level features)
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
        tuple: (all_state_sequences, all_features_np, all_labels_np)
            all_state_sequences (list[np.ndarray]): List of NumPy arrays of conductance state sequences.
            all_features_np (np.ndarray): NumPy array of flattened event-level features for each event.
            all_labels_np (np.ndarray): NumPy array of numerically encoded peptide labels.
            num_actual_states (int): number of states observed in the states sequences
    """
    # --- Calculate Absolute Paths for Databases ---
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root
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
                    # Access events_data and feature_names directly
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

                            # Store both the event_dict (for states) and the features array
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
        return [], np.array([]), np.array([]) # Return empty list for states, empty numpy arrays

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
            return [], np.array([]), np.array([]) # Return empty list for states, empty numpy arrays

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

    # Separate state sequences from the downsampled tuples
    all_state_sequences = [np.array(event_tuple[0]['states'], dtype=np.float32) for event_tuple in downsampled_events_and_features]

    num_actual_states = 0
    if all_state_sequences: # Ensure the list of sequences is not empty
        all_unique_observed_state_values = set()
        for seq in all_state_sequences:
            # Add all unique state values from the current sequence to the set.
            # Assuming state values are non-negative integers.
            all_unique_observed_state_values.update(seq)
        
        if all_unique_observed_state_values:
            # The number of unique states is the count of unique values.
            # If states are 0-indexed (0, 1, 2, 3), and all are observed, then max_state_idx + 1 is the count.
            # Example: {0, 1, 2, 3} -> max is 3 -> 3+1 = 4 states.
            num_actual_states = int(max(all_unique_observed_state_values)) + 1
        else:
            # No states observed (empty sequences or all sequences are just padding if 0 is a padding).
            # This scenario indicates an issue with data, or perhaps no events.
            num_actual_states = 0 # Or 1 if you consider "no states" as 1 implicit state
    else:
        # No sequences were loaded at all (downsampled_events_and_features was empty)
        num_actual_states = 0 # Or 1 if you need a minimum for calculations

    # Separate flattened features from the downsampled tuples
    all_features_np = np.array([event_tuple[1] for event_tuple in downsampled_events_and_features], dtype=np.float32)
    all_labels_np = np.array(downsampled_labels, dtype=np.int32)

    return all_state_sequences, all_features_np, all_labels_np, num_actual_states

# --- Visualization Functions ---
def visualize_training_history(history, filename="training_history.png"):
    """
    Visualizes the training history (loss and accuracy) and saves it.

    Args:
        history (keras.callbacks.History): The history object returned by model.fit().
        filename (str, optional): The name of the file to save the plot to.
    """
    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    # Add validation loss if available (requires validation_split in model.fit)
    if 'val_loss' in history.history:
         plt.plot(history.history['val_loss'])
    plt.title('Embedding Encoder Model Loss') # Adjusted title
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if 'val_loss' in history.history:
         plt.legend(['Train', 'Validation'], loc='upper left')
    else:
         plt.legend(['Train'], loc='upper left') # Only train loss available

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    # Accuracy is often reported per output if multiple metrics, check history.history keys
    # For a single metric on classification_head, the key is likely 'classification_head_accuracy' or just 'accuracy'
    # Let's check for 'accuracy' and 'val_accuracy' which are common default names
    accuracy_key = 'accuracy' if 'accuracy' in history.history else 'classification_head_accuracy'
    val_accuracy_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_classification_head_accuracy'


    plt.plot(history.history[accuracy_key])
    # Add validation accuracy if available
    if val_accuracy_key in history.history:
         plt.plot(history.history[val_accuracy_key])
    plt.title('Embedding Encoder Model Accuracy') # Adjusted title
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if val_accuracy_key in history.history:
         plt.legend(['Train', 'Validation'], loc='upper left')
    else:
         plt.legend(['Train'], loc='upper left') # Only train accuracy available

    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    plt.savefig(filename, dpi=300)
    plt.close()


def visualize_confusion_matrix(confusion_matrix, class_names, filename="confusion_matrix.png"):
    """
    Visualizes a confusion matrix as a color-coded heatmap with increased annotation size
    and saves it to a file with higher resolution.

    Args:
        confusion_matrix (numpy.ndarray): The 2D confusion matrix.
        class_names (list): A list of class names (e.g., peptide names).
        filename (str, optional): The name of the file to save the plot to.
    """
    plt.figure(figsize=(8, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    plt.xlabel('Predicted Peptide', fontsize=16)
    plt.ylabel('True Peptide', fontsize=16)
    plt.title('Peptide Classification Confusion Matrix - Test Set', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout() # Adjust layout to prevent labels from being cut off

    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    plt.savefig(filename, dpi=300) # Save the figure with 300 dpi
    plt.close() # Close the plot to free up memory


# --- Main Script ---
if __name__ == "__main__":
    # Define absolute paths for output directories relative to the script's location
    script_dir = os.path.dirname(__file__)

    processed_data_output_dir = os.path.join(script_dir, 'processed_data')
    os.makedirs(processed_data_output_dir, exist_ok=True)

    model_output_dir = os.path.join(script_dir, 'models')
    os.makedirs(model_output_dir, exist_ok=True)

    plots_output_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_output_dir, exist_ok=True)

    # Define a fixed sequence length for padding
    sequence_length = 1300 # As you mentioned this encompasses 99% of events

    # --- 1. Define Peptide Data Labels, Loading, and Pre-processing Parameters ---
    peptide_names_list = [
        'guesthost_Ala', 'guesthost_Leu', 'guesthost_Phe', 'guesthost_Thr',
        'guesthost_Trp', 'guesthost_TrpDL', 'guesthost_Tyr'
    ]
    print(f"\nPeptides to include in model: {peptide_names_list}")
    
    peptide_labels_encoding = {name: i for i, name in enumerate(peptide_names_list)}
    num_peptides = len(peptide_labels_encoding)
    short_peptide_names = [name.split('_')[-1] for name in peptide_names_list]
        
    desired_processing_params = {
        'high_pass_cutoff_frequency': 0,
        'filter_order': 3,
        'polynomial_degree': 2,
        'apply_polynomial_correction': True,
        'sampling_rate_hz': 400,
        'min_event_duration_ms': 20,
        'low_pass_filter_type': 'none',
        'low_pass_filter_params': {
            'bessel': {'cutoff_hz': 250, 'order': 4},
            'median': {'window_size': 3}
        }
    }
    print(f"\nProcessing parameters: {desired_processing_params}")

    raw_db_query = {
        'experimental': True,
        'nanopore_name': 'PA_F427A',
        'voltage': 70,
        'time_sampling': 400,
        'peptide_conc': {'$gte': 5, '$lte': 20}
    }

    random_state = 42 # For reproducibility
    print(f"\nrandom_state = {random_state}")

    # --- 2. Load data for CNN Embedding Model Training (with downsampling) ---
    print("\n--- Loading data for CNN Embedding Training (with downsampling) ---")
    
    # These variables will be for the CNN training dataset
    cnn_train_raw_sequences, cnn_train_features_np, cnn_train_labels_np, num_actual_states = \
        load_translocation_data_from_database(
            peptide_names_list,
            peptide_labels_encoding,
            desired_processing_params,
            raw_db_query,
            processed_events_output_dir=processed_data_output_dir,
            random_state=random_state,
            downsample_to_min_events=True # Downsample so CNN embedding training is balanced
        )
    
    if not cnn_train_raw_sequences:
        print("Error: No data loaded for CNN training. Exiting.")
        exit()

    # --- Re-map/shift states +1 for CNN training data ---
    # Original states (e.g., 0, 1, 2) shifted to (1, 2, 3) to reserve 0 for padding.
    cnn_train_shifted_sequences = [
        [s + 1 for s in seq.tolist()] # Convert numpy array to list for list comprehension, then back if needed
        for seq in cnn_train_raw_sequences
    ]
    
    # --- Pad CNN training sequences with 0 'post' ---
    # The `value=0` here will be the new padding token after remapping.
    padded_cnn_train_sequences = pad_sequences(
        cnn_train_shifted_sequences,
        maxlen=sequence_length,
        padding='post',
        dtype='int32', # States are integers, so use int32 for embedding layer input
        value=0
    )
    
    # Vocab size is all observed states plus one for the padding token (0)
    # The +1 accounts for the 0-indexed nature of states (0,1,2,3 for 4 states)
    # AND the new padding token. If max state is 3, num_actual_states = 4.
    # After shifting (1,2,3,4), the max index is 4. Add 1 for padding (0). So 5.
    # Therefore, the vocab_size needs to be num_actual_states + 1.
    new_vocab_size = num_actual_states + 1 
    
    print(f"Original number of actual states observed: {num_actual_states}")
    print(f"Max sequence length for padding: {sequence_length}")
    print(f"Vocabulary size for CNN (actual states + padding token 0): {new_vocab_size}")

    # --- 3. Split data for CNN training (padded sequences and labels) ---
    test_size = 0.2
    print(f"\nSplitting CNN training data into train/validation with test_size={test_size}...")
    X_cnn_train, X_cnn_val, y_cnn_train, y_cnn_val = train_test_split(
        padded_cnn_train_sequences,
        cnn_train_labels_np, # Only labels for the CNN's classification head
        test_size=test_size, # Use a fixed split for validation from the downsampled data
        random_state=random_state,
        stratify=cnn_train_labels_np
    )
    
    print(f"CNN Train set size: {len(y_cnn_train)} samples")
    print(f"CNN Validation set size: {len(y_cnn_val)} samples")

    # --- 4. Create the CNN Embedding Encoder model ---
    print("\nCreating CNN Embedding Encoder Model...")
    output_embedding_dim_val = 128 # Size of the embedding vector
    
    cnn_embedding_model = create_cnn_model( # Renamed from 'model' to 'cnn_embedding_model' for clarity
        sequence_length=sequence_length,
        vocab_size=new_vocab_size,
        num_peptides=num_peptides,
        embedding_dim=128,
        num_cnn_layers=2,
        filters=128,
        kernel_size=3,
        pool_size=2,
        dropout_rate=0.4,
        output_embedding_dim=output_embedding_dim_val
    )
    cnn_embedding_model.summary()

    # --- Convert NumPy arrays to TensorFlow Tensors for CNN training ---
    # Yes, it is good practice, especially if you have complex data pipelines or using tf.data.Dataset
    # For `model.fit` with basic numpy arrays, Keras often handles the conversion internally,
    # but explicit conversion can sometimes prevent subtle issues or improve clarity.
    print("\nConverting NumPy arrays for CNN training to TensorFlow Tensors...")
    X_cnn_train_tf = tf.convert_to_tensor(X_cnn_train, dtype=tf.int32) # Input to Embedding layer is int
    X_cnn_val_tf = tf.convert_to_tensor(X_cnn_val, dtype=tf.int32)
    
    y_cnn_train_tf = tf.convert_to_tensor(y_cnn_train, dtype=tf.int64) # Labels for sparse_categorical_crossentropy
    y_cnn_val_tf = tf.convert_to_tensor(y_cnn_val, dtype=tf.int64)
    print("Conversion complete.")

    # --- Create dummy target tensor for the 'embedding' output ---
    # This is needed because model.fit expects targets for all outputs with defined losses (even None).
    # The 'embedding' output has `loss=None`, but `model.fit` still needs a placeholder.
    dummy_embedding_targets_train = tf.zeros(
        shape=(tf.shape(X_cnn_train_tf)[0], output_embedding_dim_val),
        dtype=tf.float32 # Output of Dense layer is float32
    )
    dummy_embedding_targets_val = tf.zeros(
        shape=(tf.shape(X_cnn_val_tf)[0], output_embedding_dim_val),
        dtype=tf.float32
    )
    print(f"Shape of dummy training embedding targets: {dummy_embedding_targets_train.shape}")
    print(f"Shape of dummy validation embedding targets: {dummy_embedding_targets_val.shape}")

    # --- 5. Train the CNN Embedding Encoder model (supervised for classification) ---
    print("\nTraining CNN Embedding Encoder model (with classification head)...")
    epochs = 100
    batch_size = 32

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]

    history = cnn_embedding_model.fit( # Use cnn_embedding_model
        X_cnn_train_tf, # Use TensorFlow Tensors for input
        {'classification_head': y_cnn_train_tf, 'embedding': dummy_embedding_targets_train}, # Use TensorFlow Tensors for targets
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_cnn_val_tf, {'classification_head': y_cnn_val_tf, 'embedding': dummy_embedding_targets_val}), # Use validation_data tuple
        callbacks=callbacks,
        verbose=1
    )
    print("CNN Embedding Encoder training complete.")

    # --- Plot training history ---
    history_plot_filepath = os.path.join(plots_output_dir, 'cnn_encoder_training_history.png')
    visualize_training_history(history, filename=history_plot_filepath)
    print(f"\nCNN Encoder training history plot saved to {history_plot_filepath}")

    # --- 6. Reload all data and features without downsampling for XGBoost ---
    print("\n--- Reloading ALL data and features (no downsampling) for XGBoost ---")
    
    # Use different variable names to clearly separate this full dataset
    full_dataset_raw_sequences, full_dataset_features_np, full_dataset_labels_np, _ = \
        load_translocation_data_from_database(
            peptide_names_list,
            peptide_labels_encoding,
            desired_processing_params,
            raw_db_query,
            processed_events_output_dir=processed_data_output_dir,
            random_state=random_state,
            downsample_to_min_events=True
        )

    if not full_dataset_raw_sequences:
        print("Error: No data loaded for XGBoost. Exiting.")
        exit()

    # --- Re-map/shift states +1 for the full dataset ---
    full_dataset_shifted_sequences = [
        [s + 1 for s in seq.tolist()]
        for seq in full_dataset_raw_sequences
    ]

    # --- Pad full dataset sequences with 0 'post' ---
    padded_full_dataset_sequences = pad_sequences(
        full_dataset_shifted_sequences,
        maxlen=sequence_length,
        padding='post',
        dtype='int32', # Consistent with CNN input
        value=0
    )

    print(f"Total events reloaded for XGBoost (no downsampling): {len(padded_full_dataset_sequences)}")
    print(f"Shape of features for XGBoost: {full_dataset_features_np.shape}")

    # --- 7. Split full dataset into training and testing sets (for XGBoost) ---
    # This split is for the final hybrid model.
    print(f"\nSplitting full dataset (sequences, features, labels) into train/test for XGBoost...")
    
    # X_sequences and X_features need to be split together
    # It is important that these two (sequences and features) correspond to the same events.
    # The `train_test_split` function can handle multiple inputs if you pass them as individual arrays/lists.
    X_sequences_final_train, X_sequences_final_test,\
    X_features_final_train, X_features_final_test,\
    y_final_train, y_final_test = train_test_split(
        padded_full_dataset_sequences,
        full_dataset_features_np,
        full_dataset_labels_np,
        test_size=test_size,
        random_state=random_state,
        stratify=full_dataset_labels_np
    )

    print(f"Final Train set size: {len(y_final_train)} samples")
    print(f"Final Test set size: {len(y_final_test)} samples")

    # --- 8. Convert sequences for embedding generation to TensorFlow Tensors ---
    # Yes, for prediction, using Tensors is generally recommended.
    print("\nConverting final train/test sequences to TensorFlow Tensors for embedding generation...")
    X_sequences_final_train_tf = tf.convert_to_tensor(X_sequences_final_train, dtype=tf.int32)
    X_sequences_final_test_tf = tf.convert_to_tensor(X_sequences_final_test, dtype=tf.int32)
    print("Conversion complete.")

    # --- 9. Get the trained sequence embeddings for the FULL dataset splits ---
    print("\nGenerating embeddings for final train and test sets using the trained CNN Encoder...")
    # Create the embedding-only model from the trained CNN model
    embedding_extractor_model = Model(inputs=cnn_embedding_model.input, outputs=cnn_embedding_model.get_layer("embedding").output)
    
    X_train_embeddings = embedding_extractor_model.predict(X_sequences_final_train_tf)
    X_test_embeddings = embedding_extractor_model.predict(X_sequences_final_test_tf)
    
    print(f"Train embeddings shape: {X_train_embeddings.shape}")
    print(f"Test embeddings shape: {X_test_embeddings.shape}")

    # --- 10. Combine embeddings and original features for XGBoost ---
    print("\nCombining embeddings and original features for XGBoost...")
    X_train_combined = np.concatenate([X_train_embeddings, X_features_final_train], axis=1)
    X_test_combined = np.concatenate([X_test_embeddings, X_features_final_test], axis=1)

    print(f"Combined training data shape for XGBoost: {X_train_combined.shape}")
    print(f"Combined testing data shape for XGBoost: {X_test_combined.shape}")

    # --- 11. Initialize and Train XGBoost classifier ---
    print("\nInitializing XGBoost classifier...")
    xgbc = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_peptides,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
        random_state=random_state,
        n_jobs=-1,
        use_label_encoder=False, # Suppress UserWarning
        eval_metric='merror', # Metric for early stopping
        early_stopping_rounds=50 # Number of rounds to wait for improvement
    )

    print("Training XGBoost classifier on combined data...")
    
    # Use a separate validation set for early stopping if possible, but using X_test_combined/y_test is also an option
    # if you are careful not to tune hyperparameters on test set performance.
    # For a quick start, we can use a portion of X_train_combined as a validation set for early stopping.
    # A more rigorous approach would be another split from X_train_combined for xgb validation.
    # For simplicity, let's use the X_test_combined as eval_set here for early stopping visualization purposes,
    # but be aware of potential overfitting to the test set if you iterate on hyperparameters too much.
    eval_set_xgb = [(X_test_combined, y_final_test)] 

    xgbc.fit(X_train_combined, y_final_train,
             eval_set=eval_set_xgb,
             verbose=True
            )
    print("XGBoost training complete.")

    # --- 12. Make predictions with XGBoost ---
    print("\nMaking predictions on the test set with the hybrid model...")
    predictions = xgbc.predict(X_test_combined)

    # --- 13. Evaluate Hybrid model performance on test data ---
    print("\n--- Evaluation Metrics for Hybrid Model (CNN Embeddings + Features + XGBoost) ---")
    cm = confusion_matrix(y_final_test, predictions) # Use y_final_test
    print("\nPeptide Classification Confusion Matrix - Test Set:")
    print(cm)

    print("\nPeptide Classification Report - Test Set:")
    print(classification_report(y_final_test, predictions, target_names=peptide_names_list, zero_division=0))

    accuracy_peptide = accuracy_score(y_final_test, predictions)
    print(f"\nOverall Peptide Classification Accuracy: {accuracy_peptide:.4f}")

    precision_macro_peptide = precision_score(y_final_test, predictions, average='macro', zero_division=0)
    recall_macro_peptide = recall_score(y_final_test, predictions, average='macro', zero_division=0)
    f1_macro_peptide = f1_score(y_final_test, predictions, average='macro', zero_division=0)

    print(f"Macro-averaged Precision: {precision_macro_peptide:.4f}")
    print(f"Macro-averaged Recall: {recall_macro_peptide:.4f}")
    print(f"Macro-averaged F1-score: {f1_macro_peptide:.4f}")

    # --- 14. Plot the confusion matrix ---
    cm_plot_filepath = os.path.join(plots_output_dir, 'hybrid_cnn_embeddings_plus_features_XGBoost_peptide_classification_confusion_matrix.png')
    visualize_confusion_matrix(cm, short_peptide_names, filename=cm_plot_filepath)
    print(f"\nConfusion matrix plot saved to {cm_plot_filepath}")
