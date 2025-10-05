# Peptide classifier XGBoost using event-level features only
#
# B. Krantz
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score 
import pickle # Import pickle for loading data
import os # Import os to create directories

def load_peptide_data_with_features(peptide_data_paths, peptide_labels_encoding, test_size=0.2, random_state=42):
    """
    Loads translocation event features from pickle files, labels them,
    splits into training and testing sets.

    Args:
        peptide_data_paths (dict): Dictionary mapping peptide names to pickle file paths.
        peptide_labels_encoding (dict): Dictionary mapping peptide names to numerical labels (0, 1, 2...).
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random state for train/test split reproducibility.

    Returns:
        tuple: (X_train, y_train), (X_test, y_test), peptide_names_list
        X_train, X_test: Lists of translocation event features.
        y_train, y_test: true peptide labels.
        peptide_names_list: List of peptide names in order.

    """

    all_events_data = [] # List to hold dictionaries of events with features
    all_labels = [] # List of numerical encodings of peptide name labels
    all_feature_names = {} # Dictionary to hold categorized event-level and global feature names

    peptide_names_list = list(peptide_data_paths.keys())

    # load pickle files of segmented translocation event lists of features
    for peptide_name, filepath in peptide_data_paths.items():
        try:
            with open(filepath, 'rb') as infile:
                pickle_data = pickle.load(infile) # Load pickle datastructure
            events_data = pickle_data['events_data'] # Get list of dictionaries of peptide translocation events
            feature_names_dict = pickle_data['feature_names'] # Access feature names 
            labels = [peptide_labels_encoding[peptide_name]] * len(events_data)

            all_events_data.extend(events_data)
            all_labels.extend(labels)
            all_feature_names[peptide_name] = feature_names_dict
            
            print(f"Loaded {len(events_data)} translocation events with features for {peptide_name} from {filepath}")

        except FileNotFoundError:
            print(f"Error: Pickle file not found: {filepath}")
            # Decide how to handle missing files - skipping or raising an error
            # Raising an error stops execution, which might be desired if a file is essential
            # raise FileNotFoundError(f"Pickle file not found: {filepath}") # Uncomment to raise error
            continue # Skip this peptide if the file is not found
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            # Decide how to handle other loading errors
            continue # Skip this peptide if loading fails

    print(f"Total translocation event features loaded: {len(all_events_data)}")

    if not all_events_data:
        print("No data loaded. Exiting.")
        exit()

    # --- Check if event-level feature names are consistent across all peptides ---
    # Check that the feature names and elements in the lists are identical in all peptides
    # Exit if the keys don't match exactly across the loaded peptides
    # If successful then get the list of event-level feature names/keys
    event_level_feature_keys = ['event_level_scalar', 'event_level_vector_flat', 'event_level_matrix_flat']
    features_lists = [] 
    for index, peptide_name in enumerate(peptide_names_list):
        peptide_feature_names = all_feature_names[peptide_name]
        peptide_feature_names_unpacked = []
        for feature_type in event_level_feature_keys:
            peptide_feature_names_unpacked.extend(peptide_feature_names[feature_type])
        features_lists.append(peptide_feature_names_unpacked)
    # Check for mismatches by iterating through features_lists and doing set comparison
    # If no mismatches then feature name list will be successfully extracted
    mismatch_bool = any(set(list_i) != set(features_lists[0]) for list_i in features_lists)
    if mismatch_bool:
        print('Feature names were mismatched in peptide translocation event data.')
        exit()
    else:
        print('All features match in loaded peptide translocation event data.')
        event_level_feature_names = features_lists[0] # Consistent list of keys of event-level features after mismatch check is passed
    
    # Transform list of dictionaries to list of lists (features must be in consistent order)
    all_features = []
    for event in all_events_data:
        # Extract features in the defined order
        features_list = [event.get(key, 0) for key in event_level_feature_names] # Use .get() with default 0 to handle missing keys gracefully
        all_features.append(features_list)

    # Split the data into train and test sets
    # Convert lists to numpy arrays before splitting for efficiency
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(all_features),
        np.array(all_labels),
        test_size=test_size,
        random_state=random_state
    )

    return (X_train, y_train), (X_test, y_test), peptide_names_list

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
    # 1. Peptide translocation events pickle filepaths and numerical encodings
    peptide_data_paths = {
        'PeptideA': './data/peptide_A_simulated_single_channel_data_30s_with_flattened_features_min_10ms.pkl',
        'PeptideB': './data/peptide_B_simulated_single_channel_data_30s_with_flattened_features_min_10ms.pkl',
        'PeptideC': './data/peptide_C_simulated_single_channel_data_30s_with_flattened_features_min_10ms.pkl',
        'PeptideD': './data/peptide_D_simulated_single_channel_data_30s_with_flattened_features_min_10ms.pkl',
        'PeptideE': './data/peptide_E_simulated_single_channel_data_30s_with_flattened_features_min_10ms.pkl',
        'PeptideF': './data/peptide_F_simulated_single_channel_data_30s_with_flattened_features_min_10ms.pkl',
        'PeptideG': './data/peptide_G_simulated_single_channel_data_150s_with_flattened_features_min_10ms.pkl'
    }

    peptide_labels_encoding = {
        'PeptideA': 0, # For numerical labels for training
        'PeptideB': 1,
        'PeptideC': 2,
        'PeptideD': 3,
        'PeptideE': 4,
        'PeptideF': 5,
        'PeptideG': 6
    }

    # Use the dictionary keys for target names in the report, in the correct order
    ordered_peptide_names = list(peptide_labels_encoding.keys())
    # Use one-letter names for the confusion matrix plot labels if preferred
    peptide_plot_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']


    # 3. Load the data
    (X_train, y_train), (X_test, y_test), loaded_peptide_names = load_peptide_data_with_features(peptide_data_paths, peptide_labels_encoding, test_size=0.2, random_state=42)

    if not loaded_peptide_names:
        print("Data loading failed or resulted in empty datasets. Cannot proceed with training.")
        exit() # Exit if no data was loaded


    # 2. Initialize XGBoost classifier (Corrected from Regressor)
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
                             random_state=42,
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
    visualize_confusion_matrix(cm, peptide_plot_labels, filename="./plots/peptide-classifier-xgboost-confusion_matrix_min_10ms.png")
    print(f"\nConfusion matrix plot saved to ./plots/peptide-classifier-xgboost-confusion_matrix_min_10ms.png")
