# Peptide classifier TCN-Dense using conductance state sequences and event-level features
# Data are loaded from databases
# B. Krantz
import os
import numpy as np
import json
import hashlib
import random
import sys # Import sys for path manipulation
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Import TCN layers (assuming you have a tcn.py or similar module, or installed keras-tcn)
# For simplicity, I'll define a basic TCN TemporalBlock here.
# If you have the 'keras-tcn' library installed, you might prefer to use from tcn import TCN
# For this example, I'll provide a basic implementation of a TemporalBlock.
from tensorflow.keras import layers, models, Input, backend as K

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

# --- TCN Helper Block Definition ---
# This is a basic implementation of a Temporal Block, often used in TCNs.
# If you have the 'keras-tcn' library, you might replace this with its TCN layer.
def TemporalBlock(input_layer, n_filters, kernel_size, dilation_rate, dropout_rate, activation='relu'):
    # Causal Convolution 1
    conv1 = layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                          dilation_rate=dilation_rate, padding='causal',
                          kernel_initializer='he_normal')(input_layer)
    norm1 = layers.BatchNormalization()(conv1)
    act1 = layers.Activation(activation)(norm1)
    drop1 = layers.Dropout(dropout_rate)(act1)

    # Causal Convolution 2
    conv2 = layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                          dilation_rate=dilation_rate, padding='causal',
                          kernel_initializer='he_normal')(drop1)
    norm2 = layers.BatchNormalization()(conv2)
    act2 = layers.Activation(activation)(norm2)
    drop2 = layers.Dropout(dropout_rate)(act2)

    # Residual connection
    # Ensure the shortcut matches the shape of the main path
    if K.int_shape(input_layer)[-1] != n_filters:
        shortcut = layers.Conv1D(filters=n_filters, kernel_size=1, padding='same')(input_layer)
    else:
        shortcut = input_layer
    
    output = layers.add([shortcut, drop2])
    output = layers.Activation(activation)(output) # Final activation after residual connection
    return output

# --- Model Definition ---
def create_tcn_dense_model(input_sequence_length, num_features, num_peptides):
    """
    Creates a dual-input TCN-Dense model for peptide classification.
    One branch processes conductance state sequences with a TCN,
    the other processes event-level features with a Dense network.
    """
    # --- Input Branch 1: Conductance State Sequences (TCN Branch) ---
    sequence_input = Input(shape=(input_sequence_length, 1), name='state_sequence_input')
    x = layers.GaussianNoise(0.01)(sequence_input) # Add some noise for regularization

    # TCN Blocks
    # You can stack multiple TemporalBlocks with increasing dilation rates
    # Dilation rates should typically be powers of 2 (1, 2, 4, 8, ...)
    x = TemporalBlock(x, n_filters=64, kernel_size=3, dilation_rate=1, dropout_rate=0.2)
    x = TemporalBlock(x, n_filters=64, kernel_size=3, dilation_rate=2, dropout_rate=0.2)
    x = TemporalBlock(x, n_filters=128, kernel_size=3, dilation_rate=4, dropout_rate=0.3)
    x = TemporalBlock(x, n_filters=128, kernel_size=3, dilation_rate=8, dropout_rate=0.3)
    x = TemporalBlock(x, n_filters=256, kernel_size=3, dilation_rate=16, dropout_rate=0.4)
    
    # Flatten the TCN output for concatenation
    tcn_output = layers.GlobalAveragePooling1D()(x) # Or layers.Flatten()(x) if you prefer

    # --- Input Branch 2: Event-Level Features (Dense Branch) ---
    features_input = Input(shape=(num_features,), name='features_input')
    features_dense = layers.Dense(64, activation='relu')(features_input) # Increased dense layer size for features
    features_dense = layers.Dropout(0.3)(features_dense) # Add dropout to feature branch

    # --- Concatenate Both Branches ---
    concatenated = layers.concatenate([tcn_output, features_dense], name='concatenation_layer')

    # --- Classifier Head (Dense Layers) ---
    y = layers.Dense(128, activation='relu')(concatenated) # Increased head size
    y = layers.Dropout(0.5)(y) # Stronger dropout for the combined features
    output_layer = layers.Dense(num_peptides, activation='softmax', name='output_layer')(y)

    model = models.Model(inputs=[sequence_input, features_input], outputs=output_layer)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # Now returns state sequences, features, labels
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
            all_state_sequences (np.ndarray): NumPy array of conductance state sequences.
            all_features_np (np.ndarray): NumPy array of flattened event-level features for each event.
            all_labels_np (np.ndarray): NumPy array of numerically encoded peptide labels.
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
        return np.array([]), np.array([]), np.array([]) # Return empty numpy arrays

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
            return np.array([]), np.array([]), np.array([]) # Return empty numpy arrays

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

# --- Evaluation Function (for TCN-Dense) ---
def evaluate_model(model, x_states_test_padded, x_feat_test, y_test_one_hot, peptide_names_list, best_model_weights_filepath):
    """
    Evaluates the trained Keras model on the test set.

    Args:
        model: Trained Keras model.
        x_states_test_padded: Test data state sequences (NumPy array, already padded and reshaped).
        x_feat_test: Test data features (NumPy array).
        y_test_one_hot: One-hot encoded test labels (peptide classes).
        peptide_names_list: List of peptide names (for class label reporting).
        best_model_weights_filepath (str): Filepath to the best model weights (saved by ModelCheckpoint).
    """

    # 1. Load the best model weights (from ModelCheckpoint)
    model.load_weights(best_model_weights_filepath) # Load the best weights saved during training
    print(f"Loaded best model weights from: {best_model_weights_filepath}")

    # 2. Make predictions on the test set
    print("\n--- Making predictions on test set ---")
    # Pass a list of inputs to model.predict()
    y_prob_test = model.predict([x_states_test_padded, x_feat_test]) # Get probability predictions for test set
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

# --- Plotting Function (reused from previous scripts) ---
def plot_training_history(history, model_name, plot_filename):
    """
    Plots the training and validation accuracy and loss from a Keras history object.

    Args:
        history (keras.callbacks.History): The history object returned by model.fit().
        model_name (str): Name of the model for plot titles.
        plot_filename (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(plot_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Training history plot saved to: {plot_filename}")


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
        'filter_order': 3,   # This is for the existing high-pass filter
        'polynomial_degree': 2,
        'apply_polynomial_correction': True,
        'sampling_rate_hz': 400,
        'min_event_duration_ms': 20.0, # Using the best performing cutoff from previous tests
        # Selectable low-pass filtering:
        'low_pass_filter_type': 'none',   # Can be 'none' 'median' 'bessel'
        'low_pass_filter_params': {
            'bessel': {
                'cutoff_hz': 250,   # Must be less than Nyquist 0.5*sampling_rate_hz
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

    # --- 2. Load peptide translocation event datasets from the databases ---
    # This now returns state sequences, features, and labels
    all_state_sequences, all_features_np, all_labels_np = load_translocation_data_from_database(
        peptide_names_list,
        peptide_labels_encoding,
        desired_processing_params,
        raw_db_query,
        processed_events_output_dir = processed_data_output_dir,
        random_state = 42, # Using a fixed random state for reproducibility
        downsample_to_min_events = True
    )
    
    # Ensure num_features is correctly derived from the loaded data
    if all_features_np.size > 0: # Check if features array is not empty
        num_features = all_features_np.shape[1]
    else:
        print("Warning: No features loaded or all_features_np is empty. Setting num_features to 0.")
        num_features = 0
    
    # --- 3. Handle NaNs in features (impute with -1.0) ---
    if num_features > 0 and np.isnan(all_features_np).any():
        print(f"\nNaN values detected in all_features_np. Imputing with -1.0.")
        all_features_np[np.isnan(all_features_np)] = -1.0
        print("NaN imputation complete.")
        print(f"Features after imputation: Min={np.min(all_features_np):.2f}, Max={np.max(all_features_np):.2f}")
    elif num_features > 0:
        print("\nNo NaN values detected in all_features_np.")
    else:
        print("\nNo features to check for NaNs.")

    # --- 4. Feature Scaling for all_features_np ---
    if num_features > 0:
        scaler = StandardScaler()
        all_features_scaled_np = scaler.fit_transform(all_features_np)
        print(f"\nFeatures scaled using StandardScaler. Original range (pre-scaling): [{np.min(all_features_np):.2f}, {np.max(all_features_np):.2f}], Scaled range: [{np.min(all_features_scaled_np):.2f}, {np.max(all_features_scaled_np):.2f}] (approx. -3 to 3 for std normal)")
    else:
        all_features_scaled_np = all_features_np
        print("\nNo features to scale.")


    # --- 5. Train/Test Split ---
    test_size = 0.2
    # X_state_train, X_state_test for states, X_feat_train, X_feat_test for features
    X_state_train, X_state_test, \
    X_feat_train, X_feat_test, \
    y_train, y_test = train_test_split(
        all_state_sequences,     # First X input (state sequences)
        all_features_scaled_np,  # Second X input (SCALED features)
        all_labels_np,           # Y input (labels)
        test_size=test_size,
        random_state=42,
        stratify=all_labels_np
    )

    print(f"\nData split into training and testing sets:")
    print(f"  Training set: {len(X_state_train)} translocation events")
    print(f"  Testing set: {len(X_state_test)} translocation events")

    # --- 6. Determine max length from TRAINING data only (for state sequences) ---
    max_train_sequence_length = max(len(seq) for seq in X_state_train)
    print(f"Max state sequence length in training data (for padding): {max_train_sequence_length}")
    # Ensure max_effective_sequence_length is consistent with previous models
    max_effective_sequence_length = min(max_train_sequence_length, 1300) 
    print(f"Using effective max sequence length for padding: {max_effective_sequence_length}")

    # --- 7. Pad State Sequences ---
    # Pad both training and test sets to max_effective_sequence_length
    # dtype='float32' is important for neural networks
    # Padding value -1.0 as discussed
    x_states_train_padded = pad_sequences(X_state_train, maxlen=max_effective_sequence_length, padding='post', dtype='float32', value=-1.0)
    x_states_test_padded = pad_sequences(X_state_test, maxlen=max_effective_sequence_length, padding='post', dtype='float32', value=-1.0)

    # Reshape for TCN input (add a channel dimension: (num_samples, seq_length, 1))
    x_states_train_padded = np.expand_dims(x_states_train_padded, axis=-1)
    x_states_test_padded = np.expand_dims(x_states_test_padded, axis=-1)

    print(f"x_states_train_padded shape: {x_states_train_padded.shape}")
    print(f"x_states_test_padded shape: {x_states_test_padded.shape}")
    print(f"x_feat_train shape: {X_feat_train.shape}")
    print(f"x_feat_test shape: {X_feat_test.shape}")

    # --- 8. One-Hot Encode Labels ---
    num_peptides = len(peptide_labels_encoding)
    y_train_one_hot = to_categorical(y_train, num_classes=num_peptides)
    y_test_one_hot = to_categorical(y_test, num_classes=num_peptides)

    print(f"y_train_one_hot shape: {y_train_one_hot.shape}")
    print(f"y_test_one_hot shape: {y_test_one_hot.shape}")

    # Check for NaNs (should be clean after imputation)
    print(f"NaNs in x_states_train_padded: {np.isnan(x_states_train_padded).any()}")
    print(f"NaNs in x_states_test_padded: {np.isnan(x_states_test_padded).any()}")
    print(f"NaNs in x_feat_train: {np.isnan(X_feat_train).any()}")
    print(f"NaNs in x_feat_test: {np.isnan(X_feat_test).any()}")


    # --- 9. Training Callbacks ---
    model_name = "peptide_classifier_TCN_Dense_with_features_guesthost_20ms_1300_timept_max" # Descriptive model name

    best_model_weights_filepath = os.path.join(model_output_dir, f'{model_name}_best_weights.weights.h5')
    final_model_weights_filepath = os.path.join(model_output_dir, f'{model_name}_final_weights.weights.h5')
    plot_filepath = os.path.join(plots_output_dir, f'{model_name}_training_history.png')

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20, # Increased patience for deep models
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        filepath=best_model_weights_filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.000001,
        verbose=1,
        mode='min'
    )

    callbacks_list = [model_checkpoint, early_stopping, reduce_lr]

    # --- 10. Create Compiled Model ---
    model = create_tcn_dense_model(max_effective_sequence_length, num_features, num_peptides)
    model.summary()

    # --- 11. Train Model ---
    epochs = 100
    batch_size = 32

    # Using the class weights that gave the best results in the previous iteration
    class_weights = {0: 1.05, 1: 1.05, 2: 1.0, 3: 1.2, 4: 0.8, 5: 0.7, 6: 1.0}
    
    print("\n--- Starting Model Training ---")
    print(f"\nClass weights = {class_weights}") 

    history = model.fit(
        # Provide a list of inputs: [state sequences, features]
        [x_states_train_padded, X_feat_train],
        y_train_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_states_test_padded, X_feat_test], y_test_one_hot),
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1 # Set to 1 to see training progress
    )

    print("\n--- End Model Training ---")

    # --- 12. Evaluate Model on Test Set ---
    print("\n--- Evaluating Model on Test Set ---")
    evaluate_model(model, x_states_test_padded, X_feat_test, y_test_one_hot, peptide_names_list, best_model_weights_filepath)

    # --- 13. Save Model Weights and Plot Training History ---
    model.save_weights(final_model_weights_filepath)
    print(f"Trained model weights (final epoch) saved to: {final_model_weights_filepath}")
    plot_training_history(history, model_name=model_name, plot_filename=plot_filepath)

    print("\nPeptide classifier training script completed.")
