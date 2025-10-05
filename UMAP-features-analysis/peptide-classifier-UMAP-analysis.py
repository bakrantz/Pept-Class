import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import umap.umap_ as umap
import os
import sys
import json
import hashlib
import random
import matplotlib as mpl # Import matplotlib as mpl for rcParams
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans # Or HDBSCAN, DBSCAN


# Ensure proper SVG font editing capability
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12 # Default font size for general text

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

# Ensure 'plots' directory exists
plots_output_dir = './plots' # Define early for universal access
if not os.path.exists(plots_output_dir):
    os.makedirs(plots_output_dir)

# --- Data Loading Function (MODIFIED RETURN VALUE) ---
def load_translocation_data_from_database(
    peptide_names_list: list[str],
    peptide_labels_encoding: dict,
    desired_processing_params: dict,
    raw_db_query: dict = None,
    processed_events_output_dir: str = './processed_data', # Directory to save new PKL files
    random_state: int = 42,
    downsample_to_min_events: bool = True
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int, list[str]]: # Now returns state sequences, features, labels, number of states, FEATURE NAMES
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
        tuple: (all_state_sequences, all_features_np, all_labels_np, num_actual_states, feature_names_list)
            all_state_sequences (list[np.ndarray]): List of NumPy arrays of conductance state sequences.
            all_features_np (np.ndarray): NumPy array of flattened event-level features for each event.
            all_labels_np (np.ndarray): NumPy array of numerically encoded peptide labels.
            num_actual_states (int): number of states observed in the states sequences
            feature_names_list (list[str]): List of the ordered feature names that correspond to the columns in all_features_np.
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
    
    master_event_level_feature_keys = [] # Initialize here to be available even if no data

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
                        current_event_level_feature_keys = []
                        current_event_level_feature_keys.extend(feature_names_from_pkl.get('event_level_scalar', []))
                        current_event_level_feature_keys.extend(feature_names_from_pkl.get('event_level_vector_flat', []))
                        current_event_level_feature_keys.extend(feature_names_from_pkl.get('event_level_matrix_flat', []))
                        
                        # Store this list for consistency check later
                        all_event_level_feature_names_lists.append(current_event_level_feature_keys)

                        for event_dict in events_data_from_pkl:
                            # Extract features for the current event in the determined order
                            event_features_list = []
                            for key in current_event_level_feature_keys: # Use current_event_level_feature_keys
                                if key in event_dict:
                                    feature_value = event_dict[key]
                                    # Ensure feature_value is flattened if it's an array/list
                                    if isinstance(feature_value, (list, np.ndarray)):
                                        event_features_list.extend(np.array(feature_value).flatten().tolist())
                                    else:
                                        event_features_list.append(feature_value)
                                else:
                                    # Handle missing feature: append 0.0 or raise error
                                    # print(f"Warning: Feature '{key}' not found in an event dictionary for peptide '{peptide_name}'. Appending 0.0.")
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
        return [], np.array([]), np.array([]), 0, [] # Return empty list for states, empty numpy arrays, and empty feature names list

    # --- Feature Consistency Check ---
    if not all_event_level_feature_names_lists:
        # This should ideally not happen if all_events_and_features is not empty,
        # but as a safeguard.
        print("Warning: all_event_level_feature_names_lists is empty. Cannot determine feature names. Returning empty list.")
        return [], np.array([]), np.array([]), 0, []

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
            return [], np.array([]), np.array([]), 0, master_event_level_feature_keys # Return empty list for states, empty numpy arrays

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

    return all_state_sequences, all_features_np, all_labels_np, num_actual_states, master_event_level_feature_keys


# --- Main Script ---
if __name__ == "__main__":
    # Define absolute paths for output directories relative to the script's location
    script_dir = os.path.dirname(__file__)

    processed_data_output_dir = os.path.join(script_dir, 'processed_data')
    os.makedirs(processed_data_output_dir, exist_ok=True)

    model_output_dir = os.path.join(script_dir, 'models')
    os.makedirs(model_output_dir, exist_ok=True)

    # plots_output_dir is defined globally now, no need to redefine
    # os.makedirs(plots_output_dir, exist_ok=True) # Already created above

    # Define a fixed sequence length for padding (not directly used for UMAP, but good for context)
    sequence_length = 1300 # As you mentioned this encompasses 99% of events

    # --- 1. Define Peptide Data Labels, Loading, and Pre-processing Parameters ---
    peptide_names_list = [
        'guesthost_Ala', 'guesthost_Leu', 'guesthost_Phe', 'guesthost_Thr',
        'guesthost_Trp', 'guesthost_TrpDL', 'guesthost_Tyr'
    ]
    print(f"\nPeptides to include in model: {peptide_names_list}")
    
    peptide_labels_encoding = {name: i for i, name in enumerate(peptide_names_list)}
    num_peptides = len(peptide_labels_encoding)
    short_peptide_names = [name.split('_')[-1] for name in peptide_names_list] # For plotting

    # Determine the event-length filter from your table.
    selected_min_event_duration_ms = 15

    # Set Nanopore name to test
    nanopore_name = 'PA_F427Y'
    print(f"\nNanopore tested is: {nanopore_name}")
    
    desired_processing_params = {
        'high_pass_cutoff_frequency': 0,
        'filter_order': 3,
        'polynomial_degree': 2,
        'apply_polynomial_correction': True,
        'sampling_rate_hz': 400,
        'min_event_duration_ms': selected_min_event_duration_ms, # Use selected value
        'low_pass_filter_type': 'none', # Or 'bessel' or 'median' for noise testing
        'low_pass_filter_params': {
            'bessel': {'cutoff_hz': 250, 'order': 4},
            'median': {'window_size': 3}
        }
    }
    print(f"\nProcessing parameters: {desired_processing_params}")

    raw_db_query = {
        'experimental': True,
        'nanopore_name': nanopore_name,
        'voltage': 70,
        'time_sampling': 400,
        'peptide_conc': {'$gte': 5, '$lte': 20}
    }

    random_state = 42 # For reproducibility

    # 2. Load Data using the provided function (UPDATED TO RECEIVE FEATURE NAMES)
    print(f"Loading Data with min_event_duration_ms = {selected_min_event_duration_ms}...")
    sequences, features_np, labels_np, num_actual_states, feature_names_list = \
        load_translocation_data_from_database(
            peptide_names_list,
            peptide_labels_encoding,
            desired_processing_params,
            raw_db_query,
            processed_events_output_dir=processed_data_output_dir,
            random_state=random_state,
            downsample_to_min_events=True # Downsample to stay consistent for UMAP visualization
        )
    
    # Check if data was loaded
    if features_np.size == 0:
        print("No data loaded. Exiting UMAP analysis.")
        sys.exit(0)

    # --- Identify and print all-NaN features ---
    all_nan_cols_indices = np.where(np.all(np.isnan(features_np), axis=0))[0]
    if len(all_nan_cols_indices) > 0:
        print(f"\nFound {len(all_nan_cols_indices)} features that are entirely NaN:")
        dropped_feature_names = [feature_names_list[i] for i in all_nan_cols_indices]
        for idx, name in zip(all_nan_cols_indices, dropped_feature_names):
            print(f"  Index {idx}: {name}")
        
        # --- Option B: Fill all-NaN columns with 0.0 before imputation ---
        print("\nFilling all-NaN features with 0.0 before median imputation.")
        features_np_modified = features_np.copy()
        for col_idx in all_nan_cols_indices:
            features_np_modified[:, col_idx] = 0.0
        
        features_to_impute = features_np_modified
    else:
        print("\nNo features found that are entirely NaN. Proceeding with standard imputation.")
        features_to_impute = features_np.copy() # Work on a copy


    # --- NaN Imputation Step ---
    print("\nPerforming NaN imputation (median strategy) for remaining NaNs...")
    # The imputer will now only handle NaNs in columns that still have at least one non-NaN value
    imputer = SimpleImputer(strategy='median')
    features_imputed = imputer.fit_transform(features_to_impute)
    
    # Check if any column still has NaN after this step (shouldn't happen with median strategy if there's non-NaN data)
    if np.any(np.isnan(features_imputed)):
        print("Warning: NaNs still present after imputation!")
        # You can inspect specific columns:
        # nan_cols_after_imputation = np.where(np.any(np.isnan(features_imputed), axis=0))[0]
        # print(f"Columns with NaNs after imputation: {nan_cols_after_imputation}")

    print(f"Features shape after initial NaN handling and imputation: {features_imputed.shape}")
    print(f"(Original features: {features_np.shape[1]}, Imputed features: {features_imputed.shape[1]})")

    # 3. Data Preprocessing (Scaling)
    print("\nScaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)
    
    print(f"Features shape after imputation and scaling: {features_scaled.shape}")

    # 4. Apply UMAP for Dimensionality Reduction to 2D
    umap_params = {
        'n_components': 2,
        'random_state': 42,
        'n_neighbors': 15, # Controls local vs. global structure. Play with values like 10-50.
        'min_dist': 0.5,   # Controls how tightly points are clustered. Play with values like 0.0-0.5.
    }
        
    print("\nApplying UMAP to Scaled Features Data...")
    reducer = umap.UMAP(**umap_params)
    embedding = reducer.fit_transform(features_scaled)

    print(f"UMAP embedding shape: {embedding.shape}")

    # 5. Plotting the UMAP Embeddings
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2))

    # Define your custom color palette
    # Make sure the order of colors here corresponds to the order of your numerical labels (0, 1, 2...)
    # based on peptide_labels_encoding
    custom_palette_list = ["black", "red", "green", "blue", "yellow", "magenta", "cyan"]

    # If you have more peptide types than colors, Seaborn will cycle through them automatically,
    # or you can extend your palette. For now, assume len(peptide_names_list) <= len(custom_palette_list)

    # The `hue` argument should be the numerical labels_np
    # The `palette` can then be directly the list of colors in the order of your labels (0, 1, 2...)
    # Or a dictionary mapping numerical labels to colors if you want specific assignments.

    # For a direct list mapping to numerical labels:
    # We need to ensure labels_np corresponds to the order of colors in custom_palette_list
    # which it does if peptide_labels_encoding maps to 0, 1, 2... in the order of custom_palette_list.

    # Create a mapping from numerical label to actual peptide short name for the legend
    # This requires knowing the original order of peptides, which is `peptide_names_list`
    # and `short_peptide_names` corresponds to this order.

    # The `labels_np` array contains the numerical labels (0, 1, 2, ...)
    # `short_peptide_names` contains the string names in the same order as these numerical labels.

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=labels_np,  # Use the numerical labels directly for hue
        palette=custom_palette_list, # Pass the list of colors directly
        legend="full",
        alpha=0.7,
        s=15,
        ax=ax
    )

    ax.set_title(f'UMAP of Event Features for {nanopore_name}', fontsize=14)
    ax.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax.set_ylabel('UMAP Dimension 2', fontsize=14)

    # Turn on and color the axes spines
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    # Manually adjust legend labels to use short_peptide_names
    handles, plot_labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles, labels=short_peptide_names, title='Peptide', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12, frameon=False)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plot_filename = os.path.join(plots_output_dir, f'umap_peptide_features_with_{nanopore_name}_{selected_min_event_duration_ms}ms.svg')
    plt.savefig(plot_filename, format='svg', bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)

    print(f"UMAP plot saved to: {plot_filename}")

    # 6. Calculate Clustering Metrics (Example using K-Means on UMAP embedding)
    print("\nCalculating UMAP Clustering Metrics...")

    # Choose the number of clusters (n_clusters) based on your known number of peptides
    n_clusters = num_peptides # This is the number of distinct peptide types you have

    # Apply K-Means clustering to the UMAP embedding
    # Use a reproducible random_state
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    umap_clusters = kmeans_model.fit_predict(embedding)

    # Calculate metrics
    ari_score = adjusted_rand_score(labels_np, umap_clusters)
    nmi_score = normalized_mutual_info_score(labels_np, umap_clusters)

    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

    # Higher scores would indicate better agreement between the UMAP-derived clusters and the true peptide labels.
