# Peptide classifier TCN-Dense using conductance state sequences and event-level features
# Data are loaded from databases
# B. Krantz
import os
import numpy as np
import json
import hashlib
import random
import sys # Import sys for path manipulation
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Import TCN layers from keras-tcn library
from tcn import TCN # Assuming 'keras-tcn' is installed and TCN is importable
from tensorflow.keras.layers import BatchNormalization, LSTM, Dropout, GlobalAveragePooling1D, concatenate, Dense, Input
from tensorflow.keras.models import Model # Import Model directly from tensorflow.keras.models

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

# --- Data Loading Function (Modified for State Sequences) ---
def load_translocation_data_from_database(
    peptide_names_list: list[str],
    peptide_labels_encoding: dict,
    desired_processing_params: dict,
    raw_db_query: dict = None,
    processed_events_output_dir: str = './processed_data', # Directory to save new PKL files
    random_state: int = 42,
    downsample_to_min_events: bool = True
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]: # Now returns state sequences, features, labels
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

    # Separate state sequences and features from the downsampled tuples
    all_state_sequences = [np.array(event_tuple[0]['states'], dtype=np.float32) for event_tuple in downsampled_events_and_features]
    all_features_np = np.array([event_tuple[1] for event_tuple in downsampled_events_and_features], dtype=np.float32)
    all_labels_np = np.array(downsampled_labels, dtype=np.int32)

    # --- ADD THIS SECTION TO INSPECT EVENT LENGTHS ---
    print("\n--- Analyzing Event Sequence Lengths ---")
    if all_state_sequences:
        lengths = [len(seq) for seq in all_state_sequences]
        min_len = min(lengths)
        max_len = max(lengths)
        avg_len = np.mean(lengths)
        num_one_timepoint_events = sum(1 for l in lengths if l == 1) # Count events with length 1
        num_two_timepoint_events = sum(1 for l in lengths if l == 2) # Count events with length 2
        
        print(f"Total events analyzed for length: {len(lengths)}")
        print(f"Minimum event sequence length: {min_len} timepoints")
        print(f"Maximum event sequence length: {max_len} timepoints")
        print(f"Average event sequence length: {avg_len:.2f} timepoints")
        print(f"Number of 1-timepoint (2.5 ms) events found: {num_one_timepoint_events}")
        print(f"Number of 2-timepoint (5.0 ms) events found: {num_two_timepoint_events}")

        # Optional: A histogram or more detailed distribution might be useful
        # plt.figure(figsize=(10, 5))
        # plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), align='left', rwidth=0.8)
        # plt.title('Distribution of Event Sequence Lengths')
        # plt.xlabel('Event Length (timepoints)')
        # plt.ylabel('Number of Events')
        # plt.xticks(range(min(lengths), max(lengths) + 2))
        # plt.grid(axis='y', alpha=0.75)
        # plt.show() # Or save to file
    else:
        print("No state sequences available for analysis.")
    # --- END OF NEW SECTION ---
    
    return all_state_sequences, all_features_np, all_labels_np

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
        'min_event_duration_ms': 2.5, # event-length cut-off filter
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

    random_state = 42
    print(f'\nRandom state set to: {random_state}')
    
    # --- 2. Load peptide translocation event datasets from the databases ---
    # This now returns state sequences, features, and labels
    all_state_sequences, all_features_np, all_labels_np = load_translocation_data_from_database(
        peptide_names_list,
        peptide_labels_encoding,
        desired_processing_params,
        raw_db_query,
        processed_events_output_dir = './processed_data',
        random_state = random_state, # Using a fixed random state for reproducibility
        downsample_to_min_events = True
    )
    
