import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score 

def load_stream(csv_filepath):
    """
    Loads a CSV file containing raw translocation event data, extracts and scales
    the current, and extracts the state labels.

    Args:
        csv_filepath (str): The path to the input CSV file. Expected columns:
                            'Time', 'Current', 'State'.

    Returns:
        tuple: scaled_raw_current (numpy array), raw_states (numpy array)
               Returns empty arrays if the file cannot be loaded or is empty.
    """
    print(f"Loading data from {csv_filepath}...")
    try:
        # (1) Load/read csv file into pandas dataframe
        df = pd.read_csv(csv_filepath)

        # Check for expected columns
        if 'Time' not in df.columns or 'Current' not in df.columns or 'State' not in df.columns:
            print(f"Error: CSV file '{csv_filepath}' must contain 'Time', 'Current', and 'State' columns.")
            return np.array([]), np.array([])

        # (2) Extract 'Time', 'Current' and 'State' columns into numpy arrays
        # Ensure data types are suitable for calculations
        raw_times = df['Time'].values.astype(np.float32)
        raw_current = df['Current'].values.astype(np.float32)
        raw_states = df['State'].values.astype(np.int32) # Assuming states are integers

        print(f"Loaded {len(raw_current)} data points.")
        
        # (5) load_stream function returns scaled_raw_current, raw_states
        return raw_times, raw_current, raw_states

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return np.array([]), np.array([]), np.array([])
    except Exception as e:
        print(f"Error loading or processing CSV file {csv_filepath}: {e}")
        return np.array([]), np.array([]), np.array([])

def evaluate(true_states, predicted_states, target_names):
    # Calculate the confusion matrix
    cm = confusion_matrix(true_states, predicted_states)
    print("\nConductance State Classification Confusion Matrix:")
    print(cm)

    # Classification Report (Precision, Recall, F1-score per class)
    print("\nConductance State Classification Report:")
    # Pass target_names in the order of numerical labels (0 to num_classes-1)
    print(classification_report(true_states, predicted_states, target_names=target_names, zero_division=0))

    # Overall Accuracy
    accuracy_peptide = accuracy_score(true_states, predicted_states)
    print(f"\nOverall Conductance State Classification Accuracy: {accuracy_peptide:.4f}")

    # Macro-averaged Precision, Recall, F1-score
    # These metrics require average='macro' for multi-class problems
    precision_macro_peptide = precision_score(true_states, predicted_states, average='macro', zero_division=0)
    recall_macro_peptide = recall_score(true_states, predicted_states, average='macro', zero_division=0)
    f1_macro_peptide = f1_score(true_states, predicted_states, average='macro', zero_division=0)

    print(f"Macro-averaged Precision: {precision_macro_peptide:.4f}")
    print(f"Macro-averaged Recall: {recall_macro_peptide:.4f}")
    print(f"Macro-averaged F1-score: {f1_macro_peptide:.4f}")
    
if __name__ == "__main__":
    # True in this case is Clampfit assigned conductance states
    true_csv_file = "/Users/bakrantz/Documents/python/database/raw_data/PA_F427Y/guesthost_Tyr/11d05001-guesthost_Tyr-70_mV-F427Y-600_Hz-rpt_1-true.csv"

    # Predicted in this case is Python assigned conductance states
    predicted_csv_file = "/Users/bakrantz/Documents/python/database/raw_data/PA_F427Y/guesthost_Tyr/11d05001-guesthost_Tyr-70_mV-F427Y-600_Hz-rpt_1-predicted.csv"

    # Names of the conductance states
    target_names = ['B', 'I2', 'I1', 'O']

    # Load the states data
    true_times, true_current, true_states = load_stream(true_csv_file)
    predicted_times, predicted_current, predicted_states = load_stream(predicted_csv_file)

    # Evaluate the difference between the two methods
    evaluate(true_states, predicted_states, target_names)
