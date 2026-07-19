import numpy as np
import xgboost as xgb
import random 
import hashlib
import os
import sys
import json
import csv
import gc  # NEW: Required for explicit memory cleanup
import matplotlib
matplotlib.use('Agg') # Forces headless rendering to prevent Windows GUI crashes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score 
from sklearn.utils.class_weight import compute_sample_weight
import pickle
import matplotlib as mpl

# --- Global Font Settings for SVG Export ---
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12 

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
    sys.exit(1)

def load_translocation_data_from_database(
    peptide_names_list: list[str],
    peptide_labels_encoding: dict,
    desired_processing_params: dict,
    raw_db_query: dict = None,
    processed_events_output_dir: str = './processed_data', 
    random_state: int = 42,
    downsample_to_min_events: bool = False 
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads translocation event data (event-level features)
    from the PeptideEventsDatabase, processing raw data if a suitable processed
    file doesn't exist. Labels them, and optionally downsamples to equalize
    event counts per peptide.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    database_dir = os.path.join(project_root, 'database')

    raw_db_json_path = os.path.join(database_dir, 'peptide_data.json')
    processed_events_db_json_path = os.path.join(database_dir, 'peptide_events_data.json')

    print(f"Attempting to load raw database from: {raw_db_json_path}")
    print(f"Attempting to load processed events database from: {processed_events_db_json_path}")

    raw_db = PeptideDatabase(db_file=raw_db_json_path)
    processed_events_db = PeptideEventsDatabase(db_file=processed_events_db_json_path)

    all_events_and_features = []
    all_labels = []
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
            else:
                event_processor_for_new = PeptideTranslocationEvents(raw_record, desired_processing_params)
                newly_processed_record = event_processor_for_new.process_stream(
                    output_dir=processed_events_output_dir
                )

                if newly_processed_record:
                    processed_events_db.add_processed_record(newly_processed_record)
                    selected_processed_record = newly_processed_record
                else:
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
                                    event_features_list.append(np.nan) 

                            current_peptide_events_and_features.append((event_dict, np.array(event_features_list, dtype=np.float32)))
                        
                        all_events_and_features.extend(current_peptide_events_and_features)
                        all_labels.extend([peptide_labels_encoding[peptide_name]] * len(current_peptide_events_and_features))

        if not current_peptide_events_and_features:
            continue
            
    print(f"\n--- Finished Data Loading/Processing ---")
    print(f"Total translocation events loaded: {len(all_events_and_features)}")

    if not all_events_and_features:
        return np.array([]), np.array([])

    downsampled_events_and_features = []
    downsampled_labels = []

    if downsample_to_min_events:
        print("\n--- Applying Downsampling ---")
        events_and_features_by_peptide = {label: [] for label in set(all_labels)}
        for i, label in enumerate(all_labels):
            events_and_features_by_peptide[label].append(all_events_and_features[i])

        min_events = float('inf')
        actual_loaded_peptides = [p for p in peptide_names_list if peptide_labels_encoding.get(p) in events_and_features_by_peptide and len(events_and_features_by_peptide[peptide_labels_encoding[p]]) > 0]
        
        for peptide_name_for_min in actual_loaded_peptides:
            encoded_label = peptide_labels_encoding[peptide_name_for_min]
            num_events = len(events_and_features_by_peptide[encoded_label])
            if num_events < min_events:
                min_events = num_events
        
        random.seed(random_state)
        for peptide_name in peptide_names_list:
            encoded_label = peptide_labels_encoding.get(peptide_name)
            if encoded_label is not None and encoded_label in events_and_features_by_peptide:
                peptide_events_and_features = events_and_features_by_peptide[encoded_label]
                if len(peptide_events_and_features) > min_events:
                    sampled_tuples = random.sample(peptide_events_and_features, min_events)
                    downsampled_events_and_features.extend(sampled_tuples)
                    downsampled_labels.extend([encoded_label] * min_events)
                else:
                    downsampled_events_and_features.extend(peptide_events_and_features)
                    downsampled_labels.extend([encoded_label] * len(peptide_events_and_features))
    else:
        print("\n--- Downsampling is disabled. Keeping all loaded events. ---")
        downsampled_events_and_features = all_events_and_features
        downsampled_labels = all_labels

    all_features_np = np.array([event_tuple[1] for event_tuple in downsampled_events_and_features], dtype=np.float32)
    all_labels_np = np.array(downsampled_labels, dtype=np.int32)

    return all_features_np, all_labels_np

def visualize_confusion_matrix(confusion_matrix, class_names, filename="confusion_matrix.png"):
    plt.figure(figsize=(11, 10))
    
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 9}, cbar_kws={'label': 'Normalized Accuracy'})
    
    plt.xlabel('Predicted Peptide', fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('True Peptide', fontsize=14, fontweight='bold', labelpad=10)
    plt.title('Peptide Classification Confusion Matrix (Row-Normalized)', fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()

    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(filename, dpi=300)
    svg_filename = os.path.splitext(filename)[0] + ".svg"
    plt.savefig(svg_filename)

    # Free up matplotlib memory
    plt.clf()
    plt.cla()
    plt.close()

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    processed_data_output_dir = os.path.join(script_dir, 'processed_data')
    os.makedirs(processed_data_output_dir, exist_ok=True)
    plots_output_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_output_dir, exist_ok=True)

    peptide_plot_labels = [
        'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 
        'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val'
    ]

    peptide_labels_encoding = {f'guesthost_{label}': i for i, label in enumerate(peptide_plot_labels)}
    ordered_peptide_names = list(peptide_labels_encoding.keys())

    raw_db_query = {
        'experimental': True,
        'nanopore_name': 'PA',
        'voltage': 70,
        'time_sampling': 400,
        'peptide_conc': {'$gte': 5, '$lte': 20},
        'buffer': 'UBB'
    }

    # You can now safely test multiple cutoffs [25, 30, 35] without OOM crashes
    duration_cutoffs = [35]
    random_state = 42 
    summary_results = []

    # --- START OF THE LOOP ---
    for cutoff in duration_cutoffs:
        print(f"\n" + "="*60)
        print(f" RUNNING PIPELINE FOR MIN EVENT DURATION: {cutoff} ms ")
        print("="*60)

        desired_processing_params = {
            'high_pass_cutoff_frequency': 0, 
            'filter_order': 3,  
            'polynomial_degree': 2,
            'apply_polynomial_correction': True,
            'sampling_rate_hz': 400,
            'min_event_duration_ms': cutoff,  
            'low_pass_filter_type': 'none',  
            'low_pass_filter_params': {
                'bessel': {'cutoff_hz': 250, 'order': 4},
                'median': {'window_size': 3}
            }
        }
            
        all_features, all_labels = load_translocation_data_from_database(
            ordered_peptide_names,
            peptide_labels_encoding,
            desired_processing_params,
            raw_db_query,
            processed_events_output_dir = processed_data_output_dir, 
            random_state = random_state,
            downsample_to_min_events = False 
        )

        total_events_loaded = len(all_features)

        if all_features.size == 0: 
            print(f"Data loading failed or resulted in empty datasets for cutoff {cutoff}ms. Skipping to next.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            all_features, 
            all_labels,   
            test_size=0.2,
            random_state=random_state,
            stratify=all_labels 
        )

        print("\nComputing sample weights to handle class imbalance...")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        xgbc = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(ordered_peptide_names), 
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
            eval_metric='merror' 
        )

        print(f"\nTraining XGBoost classifier with balanced sample weights ({cutoff} ms)...")
        xgbc.fit(X_train, y_train, sample_weight=sample_weights)
        print("Training complete.")

        predictions = xgbc.predict(X_test)

        print(f"\n--- Evaluation Metrics for Cutoff: {cutoff} ms ---")
        cm_normalized = confusion_matrix(y_test, predictions, normalize='true')
        cm_raw = confusion_matrix(y_test, predictions)
        
        report = classification_report(
            y_test,
            predictions,
            target_names=ordered_peptide_names,
            zero_division=0
        )
        print(report)
     
        accuracy_peptide = accuracy_score(y_test, predictions)
        precision_macro_peptide = precision_score(y_test, predictions, average='macro', zero_division=0)
        recall_macro_peptide = recall_score(y_test, predictions, average='macro', zero_division=0)
        f1_macro_peptide = f1_score(y_test, predictions, average='macro', zero_division=0)

        txt_filename = os.path.join(
            plots_output_dir,
            f"peptide_classification_results_cutoff_{cutoff}ms.txt"
        )

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write(f"PEPTIDE CLASSIFICATION RESULTS (MIN EVENT DURATION = {cutoff} ms)\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Total Translocation Events Loaded: {total_events_loaded}\n\n")
            f.write(f"Accuracy          : {accuracy_peptide:.4f}\n")
            f.write(f"Macro Precision   : {precision_macro_peptide:.4f}\n")
            f.write(f"Macro Recall      : {recall_macro_peptide:.4f}\n")
            f.write(f"Macro F1-Score    : {f1_macro_peptide:.4f}\n\n")
            f.write("RAW CONFUSION MATRIX\n")
            f.write(np.array2string(cm_raw))
            f.write("\n\nNORMALIZED CONFUSION MATRIX\n")
            f.write(np.array2string(cm_normalized, precision=4))
            f.write("\n\nCLASSIFICATION REPORT\n")
            f.write(report)

        summary_results.append({
            'Cutoff (ms)': cutoff,
            'Accuracy': accuracy_peptide,
            'Macro Precision': precision_macro_peptide,
            'Macro Recall': recall_macro_peptide,
            'Macro F1-Score': f1_macro_peptide
        })

        plot_filename = f'peptide-classifier-xgboost-confusion_matrix_weighted_cutoff_{cutoff}ms.png'
        plot_filepath = os.path.join(plots_output_dir, plot_filename)
        visualize_confusion_matrix(cm_normalized, peptide_plot_labels, filename=plot_filepath)
        
        # =========================================================
        # EXPLICIT MEMORY MANAGEMENT BLOCK
        # =========================================================
        # Python's garbage collector is lazy. When looping over 550k+ event 
        # matrices (which can be gigabytes in RAM), the old arrays stay in memory 
        # while the next loop builds new ones, causing Out Of Memory (OOM) crashes.
        
        # 1. Delete large Numpy arrays and ML objects
        del all_features
        del all_labels
        del X_train
        del X_test
        del y_train
        del y_test
        del sample_weights
        del predictions
        
        # 2. Delete the XGBoost model (it holds C-level memory structures)
        del xgbc
        
        # 3. Force Python to immediately reclaim the unreferenced memory
        gc.collect()
        
    print("\nAll minimum duration cutoff loops have completed successfully.")

    # --- WRITE PERFORMANCE METRICS TO CSV FILE ---
    csv_filename = os.path.join(script_dir, 'peptide_performance_summary_30ms.csv')
    csv_fields = ['Cutoff (ms)', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score']
    
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()
            for row in summary_results:
                writer.writerow(row)
        print(f"\n[SUCCESS] Performance metrics summary exported directly to: {csv_filename}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save CSV file output: {e}")