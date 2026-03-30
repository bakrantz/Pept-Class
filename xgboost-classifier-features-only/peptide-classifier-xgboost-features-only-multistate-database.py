import numpy as np
import xgboost as xgb
import random 
import hashlib
import os
import sys
import json
import matplotlib
matplotlib.use('Agg') # Forces headless rendering to prevent Windows GUI crashes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score 
import pickle # Import pickle for loading data

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
    sys.exit(1) # Exit if essential imports fail

def load_translocation_data_from_database(
    peptide_names_list: list[str],
    peptide_labels_encoding: dict,
    desired_processing_params: dict,
    raw_db_query: dict = None,
    processed_events_output_dir: str = './processed_data', # Directory to save new PKL files
    random_state: int = 42,
    downsample_to_min_events: bool = True
) -> tuple[np.ndarray, np.ndarray]: # Return type hint to match actual return
    """
    Loads translocation event data (event-level features)
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
        tuple: (all_features_np, all_labels_np)
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
                                    # Handle missing feature: append np.nan for XGBoost to handle natively
                                    event_features_list.append(np.nan) 

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
        return np.array([]), np.array([]) # Return empty numpy arrays

    # --- Feature Consistency Check ---
    if not all_event_level_feature_names_lists:
        # This case should ideally be caught by the 'if not all_events_and_features' check above,
        # but as a safeguard, raise an error if no feature lists were collected.
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
            return np.array([]), np.array([]) # Return empty numpy arrays

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

    # Separate features from the downsampled tuples
    all_features_np = np.array([event_tuple[1] for event_tuple in downsampled_events_and_features], dtype=np.float32)
    all_labels_np = np.array(downsampled_labels, dtype=np.int32)

    return all_features_np, all_labels_np

def visualize_confusion_matrix(confusion_matrix, class_names, filename="confusion_matrix.png"):
    """
    Visualizes a confusion matrix as a color-coded heatmap with increased annotation size
    and saves it to a file with higher resolution.

    Args:
        confusion_matrix (numpy.ndarray): The 2D confusion matrix.
        class_names (list): A list of class names (e.g., peptide names).
        filename (str, optional): The name of the file to save the plot to.
                                  Defaults to "confusion_matrix.png".
    """
    plt.figure(figsize=(8, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})  # Increase annotation font size
    plt.xlabel('Predicted Peptide', fontsize=16)
    plt.ylabel('True Peptide', fontsize=16)
    plt.title('Peptide Classification Confusion Matrix - Test Set', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off

    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    plt.savefig(filename, dpi=300)  # Save the figure with 300 dpi
    plt.close() # Close the plot to free up memory

if __name__ == "__main__":
    # Define output directories
    script_dir = os.path.dirname(__file__)

    processed_data_output_dir = os.path.join(script_dir, 'processed_data')
    os.makedirs(processed_data_output_dir, exist_ok=True)

    plots_output_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_output_dir, exist_ok=True)

    # 1. Peptide numerical encodings, names and plot label abbreviations
    peptide_labels_encoding = {
        'guesthost_Ala': 0, # For numerical labels for training
        'guesthost_Leu': 1,
        'guesthost_Phe': 2,
        'guesthost_Thr': 3,
        'guesthost_Trp': 4,
        'guesthost_TrpDL': 5,
        'guesthost_Tyr': 6,
        'guesthost_His': 7
    }
    ordered_peptide_names = list(peptide_labels_encoding.keys())
    print(f"\nPeptides to include in model: {ordered_peptide_names}")

    # Use abbreviated names for the confusion matrix plot labels
    peptide_plot_labels = ['Ala', 'Leu', 'Phe', 'Thr', 'Trp', 'TrpDL', 'Tyr', 'His']

    # 2. Load from databases and split the data
    # Database query to find proper raw peptide translocation stream files
    raw_db_query = {
        'experimental': True,
        'nanopore_name': 'PA',
        'voltage': 70,
        'time_sampling': 400,
        'peptide_conc': {'$gte': 5, '$lte': 20}
        }

    # Processing parameters of various filterings to convert raw streams into translocation events
    desired_processing_params = {
        'high_pass_cutoff_frequency': 0, # Setting to zero disables high-pass filtering and baseline polynomial correction
        'filter_order': 3,  # This is for the existing high-pass filter
        'polynomial_degree': 2,
        'apply_polynomial_correction': True,
        'sampling_rate_hz': 400,
        'min_event_duration_ms': 15, # Events greater than or equal to this value will be included
        # Selectable low-pass filtering:
        'low_pass_filter_type': 'none',  # Can be 'none' 'median' 'bessel'
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
        
    random_state = 2 # Keep consistent random state for comparison
    print(f'\nRandom state: {random_state}')
    
    all_features, all_labels = load_translocation_data_from_database(
        ordered_peptide_names,
        peptide_labels_encoding,
        desired_processing_params,
        raw_db_query,
        processed_events_output_dir =  processed_data_output_dir, # Directory to save new PKL files if they are created
        random_state = random_state,
        downsample_to_min_events = True
    )

    if all_features.size == 0: # Check if the numpy array is empty
        print("Data loading failed or resulted in empty datasets. Cannot proceed with training.")
        sys.exit(1) # Use sys.exit for clean exit

    # Split the data into train and test sets
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, # all_features is already a numpy array
        all_labels,   # all_labels is already a numpy array
        test_size=test_size,
        random_state=random_state,
        stratify=all_labels # Always stratify for imbalanced classes
    )

    # 3. Initialize XGBoost classifier
    xgbc = xgb.XGBClassifier(
                             objective='multi:softmax', # Output predicted class index
                             num_class=len(ordered_peptide_names), # Number of classes
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
                             n_jobs=-1, # Use all available cores
                             eval_metric='merror' # Metric for multi-class classification error
                            )

    # 4. Train the model
    # Using early stopping can improve training efficiency
    # eval_set = [(X_train, y_train), (X_test, y_test)] # Optional: use eval_set for early stopping
    # xgbc.fit(X_train, y_train, early_stopping_rounds=50, eval_set=eval_set, verbose=True) # Use early_stopping_rounds
    print("\nTraining XGBoost classifier...")
    xgbc.fit(X_train, y_train)
    print("Training complete.")

    # 5. Make predictions
    # The 'multi:softmax' objective makes predict output the class index directly
    predictions = xgbc.predict(X_test)

    # 6. Calculate and print evaluation metrics
    print("\n--- Evaluation Metrics ---")

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("\nPeptide Classification Confusion Matrix - Test Set:")
    print(cm)

    # Classification Report (Precision, Recall, F1-score per class)
    print("\nPeptide Classification Report - Test Set:")
    # Pass target_names in the order of numerical labels (0 to num_classes-1)
    print(classification_report(y_test, predictions, target_names=ordered_peptide_names, zero_division=0))

    # Overall Accuracy
    accuracy_peptide = accuracy_score(y_test, predictions)
    print(f"\nOverall Peptide Classification Accuracy: {accuracy_peptide:.4f}")

    # Macro-averaged Precision, Recall, F1-score
    # These metrics require average='macro' for multi-class problems
    precision_macro_peptide = precision_score(y_test, predictions, average='macro', zero_division=0)
    recall_macro_peptide = recall_score(y_test, predictions, average='macro', zero_division=0)
    f1_macro_peptide = f1_score(y_test, predictions, average='macro', zero_division=0)

    print(f"Macro-averaged Precision: {precision_macro_peptide:.4f}")
    print(f"Macro-averaged Recall: {recall_macro_peptide:.4f}")
    print(f"Macro-averaged F1-score: {f1_macro_peptide:.4f}")

    # 7. Save figure of the confusion matrix
    # Pass the peptide plot labels for visualization
    plot_filepath = os.path.join(plots_output_dir, 'peptide-classifier-xgboost-confusion_matrix_5ms_event_length_cut-off.png')
    visualize_confusion_matrix(cm, peptide_plot_labels, filename=plot_filepath)
    print(f"\nConfusion matrix plot saved to {plot_filepath}")
