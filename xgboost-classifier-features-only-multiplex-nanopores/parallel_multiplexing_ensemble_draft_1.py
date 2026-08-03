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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import pickle
from typing import List, Dict, Tuple
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 0. Database Access Classes
# ==========================================
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

# ==========================================
# 1. Define the 20-Class "Periodic Table"
# ==========================================
peptides = [
    'Ala', 'Arg', 'Asn', 'Asp', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 
    'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'TrpDL', 'Tyr', 'Val'
]

peptides_plot_labels = [
    'A', 'R', 'N', 'D', 'Q', 'E', 'G', 'H', 'I', 
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'WDL', 'Y', 'V'
]

peptide_names_list = [f"guesthost_{p}" for p in peptides]
peptide_labels_encoding = {name: i for i, name in enumerate(peptide_names_list)}

# ==========================================
# 2. Define the Biophysical Property Maps
# ==========================================
aromatic_peptides = ['guesthost_Phe', 'guesthost_Trp', 'guesthost_TrpDL', 'guesthost_Tyr']
aromatic_encoding = {p: (1 if p in aromatic_peptides else 0) for p in peptide_names_list}

basic_peptides = ['guesthost_Arg', 'guesthost_Lys', 'guesthost_His']
acidic_peptides = ['guesthost_Asp', 'guesthost_Glu']
charge_encoding = {}
for p in peptide_names_list:
    if p in basic_peptides: charge_encoding[p] = 1
    elif p in acidic_peptides: charge_encoding[p] = 2
    else: charge_encoding[p] = 0

hydrophobic_peptides = [
    'guesthost_Ala', 'guesthost_Val', 'guesthost_Ile', 'guesthost_Leu', 
    'guesthost_Met', 'guesthost_Phe', 'guesthost_Tyr', 'guesthost_Trp', 
    'guesthost_TrpDL', 'guesthost_Pro'
]
hydro_encoding = {p: (1 if p in hydrophobic_peptides else 0) for p in peptide_names_list}

helix_peptides = ['guesthost_Ala', 'guesthost_Arg', 'guesthost_Gln', 'guesthost_Glu', 'guesthost_Leu', 'guesthost_Lys', 'guesthost_Met']
helix_encoding = {p: (1 if p in helix_peptides else 0) for p in peptide_names_list}

# ==========================================
# 3. Data Stream File Loading and Preprocessing
# ==========================================
def load_translocation_data_from_database(
    peptide_names_list: list[str],
    peptide_labels_encoding: dict,
    nanopore_labels_encoding: dict,
    aromatic_encoding: dict,
    charge_encoding: dict,
    hydro_encoding: dict,
    helix_encoding: dict,
    desired_processing_params: dict,
    raw_db_query: dict = None,
    processed_events_output_dir: str = './processed_data',
    random_state: int = 42,
    downsample_to_min_events: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads translocation event data from the PeptideEventsDatabase.
    Returns: features, labels, nanopore_labels, aromatic, charge, hydro, helix arrays.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    database_dir = os.path.join(project_root, 'database')
    
    raw_db = PeptideDatabase(db_file=os.path.join(database_dir, 'peptide_data.json'))
    processed_events_db = PeptideEventsDatabase(db_file=os.path.join(database_dir, 'peptide_events_data.json'))

    all_events_and_features = []
    all_labels, all_nanopore_labels, all_aromatic_labels = [], [], []
    all_charge_labels, all_hydro_labels, all_helix_labels = [], [], []

    for peptide_name in peptide_names_list:
        effective_raw_query = {'peptide_name': peptide_name}
        if raw_db_query: effective_raw_query.update(raw_db_query)
        raw_records = raw_db.retrieve_records(query=effective_raw_query)
        
        if not raw_records: continue

        current_peptide_events_and_features = []
        for raw_record in raw_records:
            nanopore_name = raw_record.nanopore_name
            params_string = json.dumps(desired_processing_params, sort_keys=True)
            param_hash = hashlib.sha256(params_string.encode('utf-8')).hexdigest()[:16]
            sanitized_peptide_name = "".join(c if c.isalnum() else '_' for c in raw_record.peptide_name).replace('__', '_').strip('_')
            expected_processed_filename = f"{sanitized_peptide_name}_{raw_record._id[:8]}_{param_hash}.pkl"

            existing = processed_events_db.retrieve_processed_records(
                query={'raw_record_id': raw_record._id, 'processed_file': expected_processed_filename}
            )

            selected_processed_record = existing[0] if existing else None
            if not selected_processed_record:
                event_processor_for_new = PeptideTranslocationEvents(raw_record, desired_processing_params)
                selected_processed_record = event_processor_for_new.process_stream(output_dir=processed_events_output_dir)
                if selected_processed_record:
                    processed_events_db.add_processed_record(selected_processed_record)
                else: continue

            pkl_filepath = processed_events_db.get_processed_file_path(selected_processed_record._id)
            event_processor = PeptideTranslocationEvents(raw_record, selected_processed_record.processing_params)
            
            if event_processor.load_events(pkl_filepath):
                events_data = event_processor.get_events_data()
                feature_names = event_processor.get_feature_names()

                if events_data and feature_names:
                    event_level_feature_keys = feature_names.get('event_level_scalar', []) + \
                                               feature_names.get('event_level_vector_flat', []) + \
                                               feature_names.get('event_level_matrix_flat', [])

                    for event_dict in events_data:
                        event_features_list = []
                        for key in event_level_feature_keys:
                            val = event_dict.get(key, np.nan)
                            if isinstance(val, (list, np.ndarray)):
                                event_features_list.extend(np.array(val).flatten().tolist())
                            else:
                                event_features_list.append(val)
                        current_peptide_events_and_features.append((event_dict, np.array(event_features_list, dtype=np.float32)))

                    num_events = len(current_peptide_events_and_features)
                    all_events_and_features.extend(current_peptide_events_and_features)
                    all_labels.extend([peptide_labels_encoding[peptide_name]] * num_events)
                    all_nanopore_labels.extend([nanopore_labels_encoding[nanopore_name]] * num_events)
                    all_aromatic_labels.extend([aromatic_encoding[peptide_name]] * num_events)
                    all_charge_labels.extend([charge_encoding[peptide_name]] * num_events)
                    all_hydro_labels.extend([hydro_encoding[peptide_name]] * num_events)
                    all_helix_labels.extend([helix_encoding[peptide_name]] * num_events)

    if not all_events_and_features:
        return (np.array([]),) * 7

    # Simple Downsampling Logic
    if downsample_to_min_events:
        print("\n--- Applying Downsampling ---")
        unique_labels = set(all_labels)
        events_by_label = {label: [] for label in unique_labels}
        for i, label in enumerate(all_labels):
            events_by_label[label].append(i)
            
        min_events = min([len(indices) for indices in events_by_label.values()])
        print(f"Downsampling to {min_events} events per class.")
        
        sampled_indices = []
        random.seed(random_state)
        for label, indices in events_by_label.items():
            sampled_indices.extend(random.sample(indices, min_events))
            
        # Rebuild lists based on sampled indices
        all_features_np = np.array([all_events_and_features[i][1] for i in sampled_indices], dtype=np.float32)
        all_labels_np = np.array([all_labels[i] for i in sampled_indices], dtype=np.int32)
        nanopore_labels_np = np.array([all_nanopore_labels[i] for i in sampled_indices], dtype=np.int32)
        aromatic_labels_np = np.array([all_aromatic_labels[i] for i in sampled_indices], dtype=np.int32)
        charge_labels_np = np.array([all_charge_labels[i] for i in sampled_indices], dtype=np.int32)
        hydro_labels_np = np.array([all_hydro_labels[i] for i in sampled_indices], dtype=np.int32)
        helix_labels_np = np.array([all_helix_labels[i] for i in sampled_indices], dtype=np.int32)
    else:
        all_features_np = np.array([e[1] for e in all_events_and_features], dtype=np.float32)
        all_labels_np = np.array(all_labels, dtype=np.int32)
        nanopore_labels_np = np.array(all_nanopore_labels, dtype=np.int32)
        aromatic_labels_np = np.array(all_aromatic_labels, dtype=np.int32)
        charge_labels_np = np.array(all_charge_labels, dtype=np.int32)
        hydro_labels_np = np.array(all_hydro_labels, dtype=np.int32)
        helix_labels_np = np.array(all_helix_labels, dtype=np.int32)            

    return all_features_np, all_labels_np, nanopore_labels_np, aromatic_labels_np, charge_labels_np, hydro_labels_np, helix_labels_np

def visualize_confusion_matrix(confusion_matrix, class_names, filename="confusion_matrix.png"):
    plt.figure(figsize=(10, 9))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10})
    plt.xlabel('Predicted Peptide', fontsize=14)
    plt.ylabel('True Peptide', fontsize=14)
    plt.title('20-Class Periodic Table Confusion Matrix', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    script_dir = os.path.dirname(__file__)
    processed_data_output_dir = os.path.join(script_dir, 'processed_data')
    plots_output_dir = os.path.join(script_dir, 'plots')
    os.makedirs(processed_data_output_dir, exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True)

    nanopore_names_for_ensemble = ['PA'] # Extend with ['PA', 'PA_F427A', 'PA_F427Y'] when multiplexing
    nanopore_labels_encoding = {name: i for i, name in enumerate(nanopore_names_for_ensemble)}

    random_state = 42
    desired_processing_params = {
        'high_pass_cutoff_frequency': 0, 'filter_order': 3, 'polynomial_degree': 2,
        'apply_polynomial_correction': True, 'sampling_rate_hz': 400, 'min_event_duration_ms': 15,
        'low_pass_filter_type': 'none', 'low_pass_filter_params': {'bessel': {'cutoff_hz': 250, 'order': 4}}
    }
    
    all_features, all_labels, all_nanopore_labels = [], [], []
    all_aromatic_labels, all_charge_labels, all_hydro_labels, all_helix_labels = [], [], [], []
    
    for nanopore_name in nanopore_names_for_ensemble:
        print(f"\n--- Loading data for {nanopore_name} nanopore ---")
        raw_db_query = {
            'experimental': True, 'nanopore_name': nanopore_name,
            'voltage': 70, 'time_sampling': 400, 'peptide_conc': {'$gte': 5, '$lte': 20}
        }

        features, labels, nanopore_labels, aromatic_labels, charge_labels, hydro_labels, helix_labels = load_translocation_data_from_database(
            peptide_names_list, peptide_labels_encoding, nanopore_labels_encoding, 
            aromatic_encoding, charge_encoding, hydro_encoding, helix_encoding,
            desired_processing_params, raw_db_query, processed_data_output_dir, random_state, downsample_to_min_events=True
        )
        
        if len(features) > 0:
            all_features.append(features); all_labels.append(labels); all_nanopore_labels.append(nanopore_labels)
            all_aromatic_labels.append(aromatic_labels); all_charge_labels.append(charge_labels)
            all_hydro_labels.append(hydro_labels); all_helix_labels.append(helix_labels)

    if len(all_features) < 1:
        print("Not enough nanopore data to perform classification. Exiting.")
        sys.exit(1)

    X_full = np.concatenate(all_features, axis=0)
    y_full = np.concatenate(all_labels, axis=0)
    nanopore_labels_full = np.concatenate(all_nanopore_labels, axis=0)
    aromatic_labels_full = np.concatenate(all_aromatic_labels, axis=0)
    charge_labels_full = np.concatenate(all_charge_labels, axis=0)
    hydro_labels_full = np.concatenate(all_hydro_labels, axis=0)
    helix_labels_full = np.concatenate(all_helix_labels, axis=0)

    combined_labels = np.array([f"{y}_{n}" for y, n in zip(y_full, nanopore_labels_full)])
    
    X_train_full, X_test_full, y_train_full, y_test_full, \
    nanopore_train, nanopore_test, aromatic_train, aromatic_test, \
    charge_train, charge_test, hydro_train, hydro_test, helix_train, helix_test = train_test_split(
        X_full, y_full, nanopore_labels_full, aromatic_labels_full, 
        charge_labels_full, hydro_labels_full, helix_labels_full,
        test_size=0.2, random_state=random_state, stratify=combined_labels
    )
    
    print("\n--- STEP 1: Training Nanopore-Specific Level 1 Property Detectors ---")
    
    # [PARICHIT - ML ARCHITECTURE NOTE]: 
    # To prevent Data Leakage in Stacking, we MUST use Out-of-Fold (OOF) predictions 
    # for the training set. If we train L1 on X_train and predict on X_train, the Meta-Classifier 
    # will overfit to falsely confident probabilities.
    
    # Dictionaries to hold final trained L1 models (for inference on test set)
    level1_classifiers = {n: {} for n in nanopore_names_for_ensemble}
    
    # Arrays to hold the clean, Out-Of-Fold probabilities for the training set
    oof_aromatic_train = np.zeros((len(X_train_full), len(nanopore_names_for_ensemble)))
    oof_charge_train = np.zeros((len(X_train_full), len(nanopore_names_for_ensemble) * 3)) # 3 classes for charge
    oof_hydro_train = np.zeros((len(X_train_full), len(nanopore_names_for_ensemble)))
    oof_helix_train = np.zeros((len(X_train_full), len(nanopore_names_for_ensemble)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    for n_idx, nanopore_name in enumerate(nanopore_names_for_ensemble):
        print(f"\n  -> Training Level 1 Specialists & Generating OOF Predictions for {nanopore_name}...")
        
        # We perform K-Fold CV specifically for this nanopore's data
        nano_mask = (nanopore_train == n_idx)
        X_train_nano = X_train_full[nano_mask]
        
        if len(X_train_nano) == 0: continue
            
        # 1. Aromaticity OOF
        xgb_arom = xgb.XGBClassifier(objective='binary:logistic', n_estimators=500, max_depth=4, random_state=random_state, n_jobs=-1)
        # 2. Charge OOF
        xgb_charge = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=500, max_depth=4, random_state=random_state, n_jobs=-1)
        # 3. Hydrophobicity OOF
        xgb_hydro = xgb.XGBClassifier(objective='binary:logistic', n_estimators=500, max_depth=4, random_state=random_state, n_jobs=-1)
        # 4. Helix Propensity OOF
        xgb_helix = xgb.XGBClassifier(objective='binary:logistic', n_estimators=500, max_depth=4, random_state=random_state, n_jobs=-1)

        # Generate OOF probabilities by splitting the X_train_full
        for train_index, val_index in skf.split(X_train_full, combined_labels):
            # Train only on the specific nanopore data WITHIN this fold
            fold_train_mask = (nanopore_train[train_index] == n_idx)
            X_fold_train = X_train_full[train_index][fold_train_mask]
            
            if len(X_fold_train) == 0: continue
                
            xgb_arom.fit(X_fold_train, aromatic_train[train_index][fold_train_mask])
            xgb_charge.fit(X_fold_train, charge_train[train_index][fold_train_mask])
            xgb_hydro.fit(X_fold_train, hydro_train[train_index][fold_train_mask])
            xgb_helix.fit(X_fold_train, helix_train[train_index][fold_train_mask])
            
            # Predict on the entire validation fold (Neutral Padding for other pores)
            oof_aromatic_train[val_index, n_idx] = xgb_arom.predict_proba(X_train_full[val_index])[:, 1]
            oof_charge_train[val_index, n_idx*3:(n_idx+1)*3] = xgb_charge.predict_proba(X_train_full[val_index])
            oof_hydro_train[val_index, n_idx] = xgb_hydro.predict_proba(X_train_full[val_index])[:, 1]
            oof_helix_train[val_index, n_idx] = xgb_helix.predict_proba(X_train_full[val_index])[:, 1]
            
        # Finally, train the "Master" L1 models on 100% of the training data to use later on the Test set
        print(f"     * Fitting Final L1 Models on full training set...")
        xgb_arom.fit(X_train_nano, aromatic_train[nano_mask]); level1_classifiers[nanopore_name]['Aromatic'] = xgb_arom
        xgb_charge.fit(X_train_nano, charge_train[nano_mask]); level1_classifiers[nanopore_name]['Charge'] = xgb_charge
        xgb_hydro.fit(X_train_nano, hydro_train[nano_mask]);   level1_classifiers[nanopore_name]['Hydro'] = xgb_hydro
        xgb_helix.fit(X_train_nano, helix_train[nano_mask]);   level1_classifiers[nanopore_name]['Helix'] = xgb_helix

    print("\n--- STEP 2: Probability Fusion for the Meta-Classifier ---")
    list_of_train_features = [X_train_full, oof_aromatic_train, oof_charge_train, oof_hydro_train, oof_helix_train]
    
    # For the Test set, we use the Master L1 models trained on the full dataset
    test_aromatic = np.zeros((len(X_test_full), len(nanopore_names_for_ensemble)))
    test_charge = np.zeros((len(X_test_full), len(nanopore_names_for_ensemble) * 3))
    test_hydro = np.zeros((len(X_test_full), len(nanopore_names_for_ensemble)))
    test_helix = np.zeros((len(X_test_full), len(nanopore_names_for_ensemble)))
    
    for n_idx, nanopore_name in enumerate(nanopore_names_for_ensemble):
        if 'Aromatic' not in level1_classifiers[nanopore_name]: continue
        test_aromatic[:, n_idx] = level1_classifiers[nanopore_name]['Aromatic'].predict_proba(X_test_full)[:, 1]
        test_charge[:, n_idx*3:(n_idx+1)*3] = level1_classifiers[nanopore_name]['Charge'].predict_proba(X_test_full)
        test_hydro[:, n_idx] = level1_classifiers[nanopore_name]['Hydro'].predict_proba(X_test_full)[:, 1]
        test_helix[:, n_idx] = level1_classifiers[nanopore_name]['Helix'].predict_proba(X_test_full)[:, 1]
        
    list_of_test_features = [X_test_full, test_aromatic, test_charge, test_hydro, test_helix]

    print("  -> Appending Sensor-ID One-Hot Flags...")
    encoder = OneHotEncoder(sparse_output=False, categories=[range(len(nanopore_names_for_ensemble))])
    nano_train_encoded = encoder.fit_transform(nanopore_train.reshape(-1, 1))
    nano_test_encoded = encoder.transform(nanopore_test.reshape(-1, 1))
    
    list_of_train_features.append(nano_train_encoded)
    list_of_test_features.append(nano_test_encoded)
    
    X_train_meta = np.hstack(list_of_train_features)
    X_test_meta = np.hstack(list_of_test_features)
    
    print("\n--- STEP 3: Training the 20-Class Master Decoder ---")
    meta_classifier = xgb.XGBClassifier(
        objective='multi:softprob', num_class=len(peptide_labels_encoding), 
        n_estimators=1000, learning_rate=0.05, max_depth=5, n_jobs=-1, random_state=random_state
    )
    meta_classifier.fit(X_train_meta, y_train_full)
    
    print("\n--- STEP 4: Evaluation and Reporting ---")
    y_pred = meta_classifier.predict(X_test_meta)
    
    cm = confusion_matrix(y_test_full, y_pred)
    acc = accuracy_score(y_test_full, y_pred)
    f1 = f1_score(y_test_full, y_pred, average='macro', zero_division=0)
    precision_macro_peptide = precision_score(y_test_full, y_pred, average='macro', zero_division=0)
    recall_macro_peptide = recall_score(y_test_full, y_pred, average='macro', zero_division=0)
    
    # FIX: Dynamically filter target names to prevent sklearn errors if a class is missing in test split
    unique_test_classes = np.unique(np.concatenate((y_test_full, y_pred)))
    target_names_test = [peptide_names_list[i] for i in unique_test_classes]

    print("\nPeptide Classification Report - Test Set:")
    print(classification_report(y_test_full, y_pred, target_names=target_names_test, zero_division=0))
    print(f"\nOverall Periodic Table Classification Accuracy: {acc:.4f}")
    print(f"Macro-averaged Precision: {precision_macro_peptide:.4f}")
    print(f"Macro-averaged Recall: {recall_macro_peptide:.4f}")
    print(f"Macro-averaged F1-score: {f1:.4f}")

    print("\n[Diagnostic] Top 10 Most Influential Features identified by Meta-Classifier:")
    importances = meta_classifier.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    for idx in top_indices:
        print(f"   Feature Index {idx}: Weight {importances[idx]:.4f}")

    plot_filepath = os.path.join(plots_output_dir, 'periodic_table_parallel_multiplex_confusion_matrix.png')
    visualize_confusion_matrix(cm, peptides_plot_labels, filename=plot_filepath)
    print(f"\nConfusion matrix plot saved to {plot_filepath}")

if __name__ == "__main__":
    main()