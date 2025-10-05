import numpy as np
import xgboost as xgb
import random
import hashlib
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import pickle
from typing import List, Dict, Tuple
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder

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
    sys.exit(1)  # Exit if essential imports fail

def load_translocation_data_from_database(
    peptide_names_list: list[str],
    peptide_labels_encoding: dict,
    nanopore_labels_encoding: dict,
    aromatic_encoding: dict,
    desired_processing_params: dict,
    raw_db_query: dict = None,
    processed_events_output_dir: str = './processed_data',
    random_state: int = 42,
    downsample_to_min_events: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads translocation event data from the PeptideEventsDatabase, processing raw data if a suitable processed
    file doesn't exist. Labels them, and optionally downsamples to equalize
    event counts per peptide. This version is modified to also return
    the nanopore labels and the aromatic/non-aromatic class labels.

    This function has been updated to no longer calculate new features on-the-fly,
    as they are now expected to be pre-calculated and stored in the .pkl files.
    """
    # --- Calculate Absolute Paths for Databases ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    database_dir = os.path.join(project_root, 'database')
    raw_db_json_path = os.path.join(database_dir, 'peptide_data.json')
    processed_events_db_json_path = os.path.join(database_dir, 'peptide_events_data.json')

    print(f"Attempting to load raw database from: {raw_db_json_path}")
    print(f"Attempting to load processed events database from: {processed_events_db_json_path}")

    # Initialize databases with the explicit, absolute file paths
    raw_db = PeptideDatabase(db_file=raw_db_json_path)
    processed_events_db = PeptideEventsDatabase(db_file=processed_events_db_json_path)

    all_events_and_features = []
    all_labels = []
    all_nanopore_labels = []
    all_aromatic_labels = []
    all_event_level_feature_names_lists = []

    print("--- Starting Data Loading/Processing from Databases ---")

    for peptide_name in peptide_names_list:
        print(f"\nSearching for data for Peptide: {peptide_name}")
        current_peptide_events_and_features = []
        effective_raw_query = {'peptide_name': peptide_name}
        if raw_db_query:
            effective_raw_query.update(raw_db_query)

        raw_records = raw_db.retrieve_records(query=effective_raw_query)
        if not raw_records:
            print(f"No raw data records found for '{peptide_name}' with query: {effective_raw_query}. Skipping.")
            continue

        print(f"Found {len(raw_records)} raw records for '{peptide_name}'.")

        for raw_record in raw_records:
            print(f"    Handling raw record: {raw_record.data_file} (ID: {raw_record._id[:8]})")

            nanopore_name = raw_record.nanopore_name

            params_string = json.dumps(desired_processing_params, sort_keys=True)
            param_hash = hashlib.sha256(params_string.encode('utf-8')).hexdigest()[:16]
            sanitized_peptide_name = "".join(c if c.isalnum() else '_' for c in raw_record.peptide_name).replace('__', '_').strip('_')
            expected_processed_filename = f"{sanitized_peptide_name}_{raw_record._id[:8]}_{param_hash}.pkl"

            existing_processed_records = processed_events_db.retrieve_processed_records(
                query={'raw_record_id': raw_record._id, 'processed_file': expected_processed_filename}
            )

            selected_processed_record = None
            if existing_processed_records:
                selected_processed_record = existing_processed_records[0]
                print(f"      Existing processed record found: {selected_processed_record.processed_file} (ID: {selected_processed_record._id[:8]})")
            else:
                print(f"      No existing processed record found with specified parameters. Initiating processing...")
                event_processor_for_new = PeptideTranslocationEvents(raw_record, desired_processing_params)
                newly_processed_record = event_processor_for_new.process_stream(output_dir=processed_events_output_dir)

                if newly_processed_record:
                    processed_events_db.add_processed_record(newly_processed_record)
                    selected_processed_record = newly_processed_record
                else:
                    print(f"      Failed to process raw record {raw_record._id}. Skipping.")
                    continue

            if selected_processed_record:
                pkl_filepath = processed_events_db.get_processed_file_path(selected_processed_record._id)
                event_processor_for_load = PeptideTranslocationEvents(raw_record, selected_processed_record.processing_params)
                if event_processor_for_load.load_events(pkl_filepath):
                    events_data_from_pkl = event_processor_for_load.get_events_data()
                    feature_names_from_pkl = event_processor_for_load.get_feature_names()

                    if events_data_from_pkl and feature_names_from_pkl:
                        event_level_feature_keys = []
                        event_level_feature_keys.extend(feature_names_from_pkl.get('event_level_scalar', []))
                        event_level_feature_keys.extend(feature_names_from_pkl.get('event_level_vector_flat', []))
                        event_level_feature_keys.extend(feature_names_from_pkl.get('event_level_matrix_flat', []))

                        # NOTE: The new features are now expected to be included in the PKL file.
                        # No need to add them manually here.
                        
                        all_event_level_feature_names_lists.append(event_level_feature_keys)

                        for event_dict in events_data_from_pkl:
                            event_features_list = []
                            for key in event_level_feature_keys:
                                if key in event_dict:
                                    feature_value = event_dict[key]
                                    if isinstance(feature_value, (list, np.ndarray)):
                                        event_features_list.extend(np.array(feature_value).flatten().tolist())
                                    else:
                                        event_features_list.append(feature_value)
                                else:
                                    # This indicates a missing feature, which should not happen if features are pre-calculated.
                                    # Instead of trying to calculate it, we'll log a warning and add a placeholder.
                                    print(f"Warning: Feature '{key}' not found in event_dict from PKL file for record {raw_record._id[:8]}. Appending NaN.")
                                    event_features_list.append(np.nan) # Handle missing features

                            current_peptide_events_and_features.append((event_dict, np.array(event_features_list, dtype=np.float32)))

                        if not current_peptide_events_and_features:
                            print(f"      PKL file {os.path.basename(pkl_filepath)} contained no event data after feature extraction. Skipping.")
                            continue

                        all_events_and_features.extend(current_peptide_events_and_features)
                        all_labels.extend([peptide_labels_encoding[peptide_name]] * len(current_peptide_events_and_features))
                        all_nanopore_labels.extend([nanopore_labels_encoding[nanopore_name]] * len(current_peptide_events_and_features))
                        # CORRECTED LINE: Using the correct aromatic_encoding dict and peptide_name
                        all_aromatic_labels.extend([aromatic_encoding[peptide_name]] * len(current_peptide_events_and_features))
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
        return np.array([]), np.array([]), np.array([]), np.array([])

    if not all_event_level_feature_names_lists:
        raise ValueError("No PKL files were loaded, cannot check feature consistency.")

    master_event_level_feature_keys = all_event_level_feature_names_lists[0]
    master_feature_set = set(master_event_level_feature_keys)

    for i, feature_set_from_pkl in enumerate(all_event_level_feature_names_lists):
        current_feature_set = set(feature_set_from_pkl)
        if current_feature_set != master_feature_set:
            raise ValueError(f"Inconsistent event-level feature names detected between PKL files. "
                             f"File at index {i} has different feature names. "
                             f"Expected set: {master_feature_set}, Found set: {current_feature_set}")
        if feature_set_from_pkl != master_event_level_feature_keys:
            print(f"Warning: Feature names are consistent but their order differs in PKL file at index {i}. "
                  f"Ensure your feature extraction logic always uses a consistent ordering of keys. "
                  f"Using the order from the first loaded PKL file for feature vector construction.")

    print(f"All {len(master_event_level_feature_keys)} event-level feature names are consistent across all loaded PKL files.")

    downsampled_events_and_features = []
    downsampled_labels = []
    downsampled_nanopore_labels = []
    downsampled_aromatic_labels = []

    if downsample_to_min_events:
        print("\n--- Applying Downsampling ---")
        events_and_features_by_peptide = {label: [] for label in set(all_labels)}
        nanopore_labels_by_peptide = {label: [] for label in set(all_labels)}
        aromatic_labels_by_peptide = {label: [] for label in set(all_labels)}

        for i, label in enumerate(all_labels):
            events_and_features_by_peptide[label].append(all_events_and_features[i])
            nanopore_labels_by_peptide[label].append(all_nanopore_labels[i])
            aromatic_labels_by_peptide[label].append(all_aromatic_labels[i])

        min_events = float('inf')
        min_peptide_name = "N/A"

        actual_loaded_peptides = [p for p in peptide_names_list if peptide_labels_encoding.get(p) in events_and_features_by_peptide and len(events_and_features_by_peptide[peptide_labels_encoding[p]]) > 0]

        if not actual_loaded_peptides:
            print("Warning: No peptides had any loaded events. Cannot downsample. Returning all loaded data (which is empty).")
            return np.array([]), np.array([]), np.array([]), np.array([])

        for peptide_name_for_min in actual_loaded_peptides:
            encoded_label = peptide_labels_encoding[peptide_name_for_min]
            num_events = len(events_and_features_by_peptide[encoded_label])
            print(f"      Peptide '{peptide_name_for_min}' (Label {encoded_label}): {num_events} events")
            if num_events < min_events:
                min_events = num_events
                min_peptide_name = peptide_name_for_min

        if min_events == 0:
            print("      Warning: One or more peptides have 0 events after initial filtering. Cannot downsample effectively. Returning all loaded data.")
            downsampled_events_and_features = all_events_and_features
            downsampled_labels = all_labels
            downsampled_nanopore_labels = all_nanopore_labels
            downsampled_aromatic_labels = all_aromatic_labels
        else:
            print(f"      Downsampling all peptides to {min_events} events based on '{min_peptide_name}'.")
            random.seed(random_state)
            for peptide_name in peptide_names_list:
                encoded_label = peptide_labels_encoding.get(peptide_name)
                if encoded_label is not None and encoded_label in events_and_features_by_peptide:
                    peptide_events_and_features = events_and_features_by_peptide[encoded_label]
                    peptide_nanopore_labels = nanopore_labels_by_peptide[encoded_label]
                    peptide_aromatic_labels = aromatic_labels_by_peptide[encoded_label]

                    # Combine features, nanopore labels, and aromatic labels for consistent sampling
                    combined_data = list(zip(peptide_events_and_features, peptide_nanopore_labels, peptide_aromatic_labels))
                    sampled_combined = random.sample(combined_data, min_events)

                    sampled_features, sampled_nanopore_labels, sampled_aromatic_labels = zip(*sampled_combined) if sampled_combined else ([], [], [])

                    downsampled_events_and_features.extend(sampled_features)
                    downsampled_labels.extend([encoded_label] * min_events)
                    downsampled_nanopore_labels.extend(sampled_nanopore_labels)
                    downsampled_aromatic_labels.extend(sampled_aromatic_labels)

                    print(f"      Peptide '{peptide_name}' (Label {encoded_label}) downsampled from {len(peptide_events_and_features)} to {min_events} events.")
                else:
                    print(f"      Skipping downsampling for '{peptide_name}', no data found.")
    else:
        print("\n--- Downsampling is disabled. Keeping all loaded events. ---")
        downsampled_events_and_features = all_events_and_features
        downsampled_labels = all_labels
        downsampled_nanopore_labels = all_nanopore_labels
        downsampled_aromatic_labels = all_aromatic_labels

    print(f"Total translocation events after downsampling: {len(downsampled_events_and_features)}")

    all_features_np = np.array([event_tuple[1] for event_tuple in downsampled_events_and_features], dtype=np.float32)
    all_labels_np = np.array(downsampled_labels, dtype=np.int32)
    nanopore_labels_np = np.array(downsampled_nanopore_labels, dtype=np.int32)
    aromatic_labels_np = np.array(downsampled_aromatic_labels, dtype=np.int32)

    return all_features_np, all_labels_np, nanopore_labels_np, aromatic_labels_np


    
def visualize_confusion_matrix(confusion_matrix, class_names, filename="confusion_matrix.png"):
    """
    Visualizes a confusion matrix as a color-coded heatmap with increased annotation size
    and saves it to a file with higher resolution.
    """
    plt.figure(figsize=(8, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    plt.xlabel('Predicted Peptide', fontsize=16)
    plt.ylabel('True Peptide', fontsize=16)
    plt.title('Peptide Classification Confusion Matrix - Test Set', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Define output directories
    script_dir = os.path.dirname(__file__)
    processed_data_output_dir = os.path.join(script_dir, 'processed_data')
    os.makedirs(processed_data_output_dir, exist_ok=True)
    plots_output_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_output_dir, exist_ok=True)

    # 1. Define nanopore, peptide, and aromatic numerical encodings and names
    nanopore_labels_encoding = {'PA': 0, 'PA_F427A': 1}
    nanopore_names_for_ensemble = ['PA', 'PA_F427A']

    peptide_labels_encoding = {
        'guesthost_Ala': 0, 'guesthost_Leu': 1, 'guesthost_Phe': 2,
        'guesthost_Thr': 3, 'guesthost_Trp': 4, 'guesthost_TrpDL': 5,
        'guesthost_Tyr': 6
    }
    ordered_peptide_names = list(peptide_labels_encoding.keys())
    peptide_plot_labels = ['Ala', 'Leu', 'Phe', 'Thr', 'Trp', 'TrpDL', 'Tyr']

    aromatic_peptides = ['guesthost_Phe', 'guesthost_Trp', 'guesthost_TrpDL', 'guesthost_Tyr']
    non_aromatic_peptides = ['guesthost_Ala', 'guesthost_Leu', 'guesthost_Thr']
    
    aromatic_encoding = {
        'guesthost_Phe': 1, 'guesthost_Trp': 1, 'guesthost_TrpDL': 1, 'guesthost_Tyr': 1,
        'guesthost_Ala': 0, 'guesthost_Leu': 0, 'guesthost_Thr': 0
    }
    
    # 2. Load data from the database
    random_state = 42
    print(f'\nRandom state: {random_state}')
    desired_processing_params = {
        'high_pass_cutoff_frequency': 0,
        'filter_order': 3,
        'polynomial_degree': 2,
        'apply_polynomial_correction': True,
        'sampling_rate_hz': 400,
        'min_event_duration_ms': 15,
        'low_pass_filter_type': 'none',
        'low_pass_filter_params': {'bessel': {'cutoff_hz': 250, 'order': 4}, 'median': {'window_size': 3}}
    }
    
    all_features = []
    all_labels = []
    all_nanopore_labels = []
    all_aromatic_labels = []
    
    for nanopore_name in nanopore_names_for_ensemble:
        print(f"\n--- Loading data for {nanopore_name} nanopore ---")
        raw_db_query = {
            'experimental': True,
            'nanopore_name': nanopore_name,
            'voltage': 70,
            'time_sampling': 400,
            'peptide_conc': {'$gte': 5, '$lte': 20}
        }
        features, labels, nanopore_labels, aromatic_labels = load_translocation_data_from_database(
            ordered_peptide_names, peptide_labels_encoding, nanopore_labels_encoding, aromatic_encoding,
            desired_processing_params, raw_db_query, processed_data_output_dir, random_state, downsample_to_min_events=True
        )
        if len(features) > 0:
            all_features.append(features)
            all_labels.append(labels)
            all_nanopore_labels.append(nanopore_labels)
            all_aromatic_labels.append(aromatic_labels)
        else:
            print(f"No data loaded for {nanopore_name}. Skipping.")

    if len(all_features) < 2:
        print("Not enough nanopore data to perform classification. Exiting.")
        sys.exit(1)

    # Concatenate all data for a single global split
    X_full = np.concatenate(all_features, axis=0)
    y_full = np.concatenate(all_labels, axis=0)
    nanopore_labels_full = np.concatenate(all_nanopore_labels, axis=0)
    aromatic_labels_full = np.concatenate(all_aromatic_labels, axis=0)

    # Perform a single global train-test split, stratifying by both peptide and nanopore type
    # A custom split is needed to handle stratification by two columns.
    # The easiest way is to create a combined label.
    combined_labels = np.array([f"{y}_{n}" for y, n in zip(y_full, nanopore_labels_full)])
    
    X_train_full, X_test_full, y_train_full, y_test_full, \
    nanopore_train, nanopore_test, aromatic_train, aromatic_test = train_test_split(
        X_full, y_full, nanopore_labels_full, aromatic_labels_full,
        test_size=0.2, random_state=random_state, stratify=combined_labels
    )
    
    print(f"\nTotal training events: {len(y_train_full)}")
    print(f"Total testing events: {len(y_test_full)}")
    
    # 3. Create subsets for training Level 1 classifiers
    # Level 1 Aromatic/Non-Aromatic Classifiers (one per nanopore)
    level1_classifiers = {}
    for nanopore_label, nanopore_name in nanopore_labels_encoding.items():
        # Create a training set for this specific nanopore
        is_nanopore_train = (nanopore_train == nanopore_name)
        X_train_nanopore = X_train_full[is_nanopore_train]
        y_train_aromatic = aromatic_train[is_nanopore_train]
        
        if len(np.unique(y_train_aromatic)) < 2:
            print(f"Skipping Level 1 Aromatic/Non-Aromatic classifier for {nanopore_label} "
                  f"due to insufficient classes in training data.")
            continue
            
        print(f"\n--- STEP 1: Training Level 1 Aromatic/Non-Aromatic Classifier for {nanopore_label} nanopore ---")
        xgbc_level1 = xgb.XGBClassifier(
            objective='binary:logistic', n_estimators=1000, learning_rate=0.05,
            max_depth=5, random_state=random_state, n_jobs=-1, eval_metric='logloss'
        )
        xgbc_level1.fit(X_train_nanopore, y_train_aromatic)
        level1_classifiers[nanopore_label] = xgbc_level1
        print(f"Level 1 {nanopore_label} classifier trained successfully.")


    # 4. Augment data with Level 1 probabilities and train Level 2 classifiers
    # Now, Level 2 classifiers are split by Aromatic/Non-Aromatic AND nanopore type
    level2_classifiers = {}
    
    print("\n--- STEP 2: Augmenting data with Level 1 probabilities and training Level 2 classifiers ---")
    
    # Generate Level 1 predictions for the entire training set
    level1_preds_train = np.empty((X_train_full.shape[0], len(nanopore_labels_encoding)))
    for i, nanopore_label in enumerate(nanopore_labels_encoding.keys()):
        if nanopore_label in level1_classifiers:
            level1_preds_train[:, i] = level1_classifiers[nanopore_label].predict_proba(X_train_full)[:, 1]
        else:
            # Handle cases where a level 1 classifier wasn't trained
            level1_preds_train[:, i] = 0.5 # Neutral prediction

    # Concatenate Level 1 probabilities to the original feature set
    X_train_aug = np.hstack((X_train_full, level1_preds_train))

    # Split the augmented training data for Level 2 classifiers
    aromatic_train_indices = np.where(aromatic_train == 1)[0]
    non_aromatic_train_indices = np.where(aromatic_train == 0)[0]
    
    # Train Level 2 Aromatic classifier
    X_train_aromatic = X_train_aug[aromatic_train_indices]
    y_train_aromatic_peptide = y_train_full[aromatic_train_indices]
    
    aromatic_peptide_indices = [peptide_labels_encoding[p] for p in aromatic_peptides]
    y_train_aromatic_peptide_filtered = y_train_aromatic_peptide[np.isin(y_train_aromatic_peptide, aromatic_peptide_indices)]
    X_train_aromatic_filtered = X_train_aromatic[np.isin(y_train_aromatic_peptide, aromatic_peptide_indices)]
    
    unique_aromatic_classes = np.unique(y_train_aromatic_peptide_filtered)
    
    if len(unique_aromatic_classes) > 1:
        # Create a local label encoding for the aromatic classes
        aromatic_global_to_local_map = {global_label: local_label for local_label, global_label in enumerate(unique_aromatic_classes)}
        aromatic_local_to_global_map = {local_label: global_label for global_label, local_label in aromatic_global_to_local_map.items()}
        y_train_aromatic_peptide_local = np.array([aromatic_global_to_local_map[y] for y in y_train_aromatic_peptide_filtered])
        
        print("Training Level 2 Aromatic classifier.")
        xgbc_level2_aromatic = xgb.XGBClassifier(
            objective='multi:softmax', num_class=len(unique_aromatic_classes),
            n_estimators=1000, learning_rate=0.05, max_depth=5,
            random_state=random_state, n_jobs=-1, eval_metric='merror'
        )
        xgbc_level2_aromatic.fit(X_train_aromatic_filtered, y_train_aromatic_peptide_local)
        level2_classifiers['aromatic'] = (xgbc_level2_aromatic, aromatic_local_to_global_map)
        print("Level 2 Aromatic classifier trained successfully.")
    else:
        print("Skipping Level 2 Aromatic classifier due to insufficient classes.")

    # Train Level 2 Non-Aromatic classifier
    X_train_non_aromatic = X_train_aug[non_aromatic_train_indices]
    y_train_non_aromatic_peptide = y_train_full[non_aromatic_train_indices]

    non_aromatic_peptide_indices = [peptide_labels_encoding[p] for p in non_aromatic_peptides]
    y_train_non_aromatic_peptide_filtered = y_train_non_aromatic_peptide[np.isin(y_train_non_aromatic_peptide, non_aromatic_peptide_indices)]
    X_train_non_aromatic_filtered = X_train_non_aromatic[np.isin(y_train_non_aromatic_peptide, non_aromatic_peptide_indices)]

    unique_non_aromatic_classes = np.unique(y_train_non_aromatic_peptide_filtered)

    if len(unique_non_aromatic_classes) > 1:
        # Create a local label encoding for the non-aromatic classes
        non_aromatic_global_to_local_map = {global_label: local_label for local_label, global_label in enumerate(unique_non_aromatic_classes)}
        non_aromatic_local_to_global_map = {local_label: global_label for global_label, local_label in non_aromatic_global_to_local_map.items()}
        y_train_non_aromatic_peptide_local = np.array([non_aromatic_global_to_local_map[y] for y in y_train_non_aromatic_peptide_filtered])
        
        print("Training Level 2 Non-Aromatic classifier.")
        xgbc_level2_non_aromatic = xgb.XGBClassifier(
            objective='multi:softmax', num_class=len(unique_non_aromatic_classes),
            n_estimators=1000, learning_rate=0.05, max_depth=5,
            random_state=random_state, n_jobs=-1, eval_metric='merror'
        )
        xgbc_level2_non_aromatic.fit(X_train_non_aromatic_filtered, y_train_non_aromatic_peptide_local)
        level2_classifiers['non_aromatic'] = (xgbc_level2_non_aromatic, non_aromatic_local_to_global_map)
        print("Level 2 Non-Aromatic classifier trained successfully.")
    else:
        print("Skipping Level 2 Non-Aromatic classifier due to insufficient classes.")
    
    # 5. Training a meta-classifier for final predictions
    print("\n--- STEP 3: Training a meta-classifier for final predictions ---")
    
    # Generate Level 2 predictions for the entire training set
    level2_preds_train = np.zeros((X_train_full.shape[0], len(peptide_labels_encoding)))
    
    if 'aromatic' in level2_classifiers:
        xgbc_level2_aromatic, aromatic_local_to_global_map = level2_classifiers['aromatic']
        X_train_aromatic = X_train_aug[aromatic_train_indices]
        y_pred_proba_aromatic = xgbc_level2_aromatic.predict_proba(X_train_aromatic)
        # Map the Level 2 predicted classes back to their original global labels
        for local_class_idx, global_class_idx in aromatic_local_to_global_map.items():
            level2_preds_train[aromatic_train_indices, global_class_idx] = y_pred_proba_aromatic[:, local_class_idx]

    if 'non_aromatic' in level2_classifiers:
        xgbc_level2_non_aromatic, non_aromatic_local_to_global_map = level2_classifiers['non_aromatic']
        X_train_non_aromatic = X_train_aug[non_aromatic_train_indices]
        y_pred_proba_non_aromatic = xgbc_level2_non_aromatic.predict_proba(X_train_non_aromatic)
        for local_class_idx, global_class_idx in non_aromatic_local_to_global_map.items():
            level2_preds_train[non_aromatic_train_indices, global_class_idx] = y_pred_proba_non_aromatic[:, local_class_idx]

    # Combine Level 1 and Level 2 predictions for the meta-classifier
    X_train_meta_blended = np.hstack((level1_preds_train, level2_preds_train))
    
    meta_classifier = xgb.XGBClassifier(
        objective='multi:softprob', num_class=len(peptide_labels_encoding),
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        random_state=random_state, n_jobs=-1, eval_metric='mlogloss'
    )
    meta_classifier.fit(X_train_meta_blended, y_train_full)
    print("Meta-classifier trained successfully.")

    # 6. Feature Importance Analysis for the Meta-Classifier
    print("\n--- STEP 4: Feature Importance Analysis for the Meta-Classifier ---")
    # The features for the meta-classifier are the probabilistic outputs from the Level 1 and Level 2 classifiers.
    # Create descriptive names for these features.
    
    # Level 1 features (Aromatic/Non-Aromatic probability for each nanopore)
    level1_feature_names = [f"Level1_{nanopore_label}_proba" for nanopore_label in nanopore_labels_encoding.keys()]
    
    # Level 2 features (Peptide-specific probabilities for all classes)
    level2_feature_names = [f"Level2_{peptide_name}_proba" for peptide_name in ordered_peptide_names]
    
    # Combine all feature names
    meta_classifier_feature_names = level1_feature_names + level2_feature_names

    # Get the feature importances from the trained meta-classifier
    feature_importances = meta_classifier.feature_importances_
    
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': meta_classifier_feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nMeta-Classifier Feature Importance:")
    print(importance_df)
    
    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Meta-Classifier Feature Importance')
    plt.tight_layout()
    importance_plot_filepath = os.path.join(plots_output_dir, 'meta_classifier_feature_importance.png')
    plt.savefig(importance_plot_filepath)
    print(f"Feature importance plot saved to {importance_plot_filepath}")
    plt.close()


    # 7. Evaluation on the test set
    print("\n--- Evaluation Metrics for the Probabilistic Blending Ensemble ---")
    
    # Process the test data through the same pipeline
    level1_preds_test = np.empty((X_test_full.shape[0], len(nanopore_labels_encoding)))
    for i, nanopore_label in enumerate(nanopore_labels_encoding.keys()):
        if nanopore_label in level1_classifiers:
            level1_preds_test[:, i] = level1_classifiers[nanopore_label].predict_proba(X_test_full)[:, 1]
        else:
            level1_preds_test[:, i] = 0.5
            
    X_test_aug = np.hstack((X_test_full, level1_preds_test))
    
    level2_preds_test = np.zeros((X_test_full.shape[0], len(peptide_labels_encoding)))
    
    # Get indices for aromatic and non-aromatic peptides in the test set
    aromatic_test_indices = np.where(aromatic_test == 1)[0]
    non_aromatic_test_indices = np.where(aromatic_test == 0)[0]

    if 'aromatic' in level2_classifiers:
        xgbc_level2_aromatic, aromatic_local_to_global_map = level2_classifiers['aromatic']
        X_test_aromatic = X_test_aug[aromatic_test_indices]
        y_pred_proba_aromatic_test = xgbc_level2_aromatic.predict_proba(X_test_aromatic)
        for local_class_idx, global_class_idx in aromatic_local_to_global_map.items():
            level2_preds_test[aromatic_test_indices, global_class_idx] = y_pred_proba_aromatic_test[:, local_class_idx]
            
    if 'non_aromatic' in level2_classifiers:
        xgbc_level2_non_aromatic, non_aromatic_local_to_global_map = level2_classifiers['non_aromatic']
        X_test_non_aromatic = X_test_aug[non_aromatic_test_indices]
        y_pred_proba_non_aromatic_test = xgbc_level2_non_aromatic.predict_proba(X_test_non_aromatic)
        for local_class_idx, global_class_idx in non_aromatic_local_to_global_map.items():
            level2_preds_test[non_aromatic_test_indices, global_class_idx] = y_pred_proba_non_aromatic_test[:, local_class_idx]
            
    X_test_meta_blended = np.hstack((level1_preds_test, level2_preds_test))
    
    final_predictions = meta_classifier.predict(X_test_meta_blended)
    
    cm = confusion_matrix(y_test_full, final_predictions)
    print("\nPeptide Classification Confusion Matrix - Test Set:")
    print(cm)
    print("\nPeptide Classification Report - Test Set:")
    print(classification_report(y_test_full, final_predictions, target_names=ordered_peptide_names, zero_division=0))
    accuracy_peptide = accuracy_score(y_test_full, final_predictions)
    print(f"\nOverall Peptide Classification Accuracy: {accuracy_peptide:.4f}")
    precision_macro_peptide = precision_score(y_test_full, final_predictions, average='macro', zero_division=0)
    recall_macro_peptide = recall_score(y_test_full, final_predictions, average='macro', zero_division=0)
    f1_macro_peptide = f1_score(y_test_full, final_predictions, average='macro', zero_division=0)

    print(f"Macro-averaged Precision: {precision_macro_peptide:.4f}")
    print(f"Macro-averaged Recall: {recall_macro_peptide:.4f}")
    print(f"Macro-averaged F1-score: {f1_macro_peptide:.4f}")

    # Save figure of the confusion matrix for the final blending model
    plot_filepath = os.path.join(plots_output_dir, 'peptide-classifier-xgboost-hierarchical-blending-nanopore-specialist-enhanced-confusion_matrix.png')
    visualize_confusion_matrix(cm, peptide_plot_labels, filename=plot_filepath)
    print(f"\nConfusion matrix plot saved to {plot_filepath}")
