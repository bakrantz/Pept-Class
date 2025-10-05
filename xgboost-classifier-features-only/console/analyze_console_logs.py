import os
from docx import Document
import re
import numpy as np

def extract_metrics_from_docx(docx_path):
    """
    Extracts overall accuracy, macro-averaged precision, recall, and F1-score
    from a .docx document containing a scikit-learn classification report.
    """
    document = Document(docx_path)
    full_text = []
    for para in document.paragraphs:
        full_text.append(para.text)
    
    full_text_str = "\n".join(full_text)

    metrics = {}

    # Extract Overall Peptide Classification Accuracy
    accuracy_match = re.search(r"Overall Peptide Classification Accuracy: (\d+\.\d+)", full_text_str)
    if accuracy_match:
        metrics['accuracy'] = float(accuracy_match.group(1))

    # Extract Macro-averaged Precision, Recall, F1-score
    macro_precision_match = re.search(r"Macro-averaged Precision:\s*(\d+\.\d+)", full_text_str)
    if macro_precision_match:
        metrics['macro_precision'] = float(macro_precision_match.group(1))

    macro_recall_match = re.search(r"Macro-averaged Recall:\s*(\d+\.\d+)", full_text_str)
    if macro_recall_match:
        metrics['macro_recall'] = float(macro_recall_match.group(1))

    macro_f1_match = re.search(r"Macro-averaged F1-score:\s*(\d+\.\d+)", full_text_str)
    if macro_f1_match:
        metrics['macro_f1'] = float(macro_f1_match.group(1))
    
    return metrics

def find_best_replicate(replicate_data, metric_key='macro_f1'):
    """
    Finds the replicate with the highest value for the specified metric_key.

    Args:
        replicate_data (list of dict): A list where each dict contains
                                       {'filename': str, 'metrics': dict}.
        metric_key (str): The key in the 'metrics' dict to use for comparison
                          (e.g., 'accuracy', 'macro_f1').

    Returns:
        tuple: (best_filename, best_metric_value)
               Returns (None, None) if replicate_data is empty or no valid metrics.
    """
    best_filename = None
    best_metric_value = -1.0 # Initialize with a low value

    for entry in replicate_data:
        filename = entry['filename']
        metrics = entry['metrics']
        
        if metric_key in metrics and metrics[metric_key] is not None:
            current_value = metrics[metric_key]
            if current_value > best_metric_value:
                best_metric_value = current_value
                best_filename = filename
    
    return best_filename, best_metric_value


def main():
    # List of your DOCX files for WT
    docx_files_WT = [
        "console-xgboost-simple-WT-15ms-rpt1.docx",
        "console-xgboost-simple-WT-15ms-rpt2.docx",
        "console-xgboost-simple-WT-15ms-rpt3.docx",
        "console-xgboost-simple-WT-15ms-rpt4.docx",
        "console-xgboost-simple-WT-15ms-rpt5.docx",
    ]

    # List of your DOCX files for F427A
    docx_files_F427A = [
        "console-xgboost-simple-F427A-15ms-rpt1.docx",
        "console-xgboost-simple-F427A-15ms-rpt2.docx",
        "console-xgboost-simple-F427A-15ms-rpt3.docx",
        "console-xgboost-simple-F427A-15ms-rpt4.docx",
        "console-xgboost-simple-F427A-15ms-rpt5.docx",
    ]

    # List of your DOCX files for F427Y
    docx_files_F427Y = [
        "console-xgboost-simple-F427Y-15ms-rpt1.docx",
        "console-xgboost-simple-F427Y-15ms-rpt2.docx",
        "console-xgboost-simple-F427Y-15ms-rpt3.docx",
        "console-xgboost-simple-F427Y-15ms-rpt4.docx",
        "console-xgboost-simple-F427Y-15ms-rpt5.docx",
    ]
        
    # Directory where your DOCX files are located (optional, adjust if files are elsewhere)
    docx_directory = "." 

    all_accuracies = []
    all_macro_precisions = []
    all_macro_recalls = []
    all_macro_f1s = []
    
    replicates_data = [] # To store filename and metrics for best selection

    print("--- Extracting Metrics ---")
    for filename in docx_files_F427Y:
        filepath = os.path.join(docx_directory, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File not found - {filepath}. Skipping.")
            continue

        print(f"Processing {filename}...")
        metrics = extract_metrics_from_docx(filepath)

        if metrics:
            all_accuracies.append(metrics.get('accuracy'))
            all_macro_precisions.append(metrics.get('macro_precision'))
            all_macro_recalls.append(metrics.get('macro_recall'))
            all_macro_f1s.append(metrics.get('macro_f1'))
            print(f"  Extracted: Acc={metrics.get('accuracy')}, P={metrics.get('macro_precision')}, R={metrics.get('macro_recall')}, F1={metrics.get('macro_f1')}")
            replicates_data.append({'filename': filename, 'metrics': metrics})
        else:
            print(f"  No metrics found in {filename}.")
    
    print("\n--- Summary Statistics ---")

    # Filter out None values in case some metrics weren't found
    all_accuracies = [x for x in all_accuracies if x is not None]
    all_macro_precisions = [x for x in all_macro_precisions if x is not None]
    all_macro_recalls = [x for x in all_macro_recalls if x is not None]
    all_macro_f1s = [x for x in all_macro_f1s if x is not None]

    if all_accuracies:
        print(f"Overall Accuracy (N={len(all_accuracies)}):")
        print(f"  Mean: {np.mean(all_accuracies):.4f}")
        print(f"  Std Dev: {np.std(all_accuracies):.4f}")
    else:
        print("No overall accuracy data to process.")

    if all_macro_precisions:
        print(f"\nMacro-averaged Precision (N={len(all_macro_precisions)}):")
        print(f"  Mean: {np.mean(all_macro_precisions):.4f}")
        print(f"  Std Dev: {np.std(all_macro_precisions):.4f}")
    else:
        print("No macro-averaged precision data to process.")

    if all_macro_recalls:
        print(f"\nMacro-averaged Recall (N={len(all_macro_recalls)}):")
        print(f"  Mean: {np.mean(all_macro_recalls):.4f}")
        print(f"  Std Dev: {np.std(all_macro_recalls):.4f}")
    else:
        print("No macro-averaged recall data to process.")

    if all_macro_f1s:
        print(f"\nMacro-averaged F1-score (N={len(all_macro_f1s)}):")
        print(f"  Mean: {np.mean(all_macro_f1s):.4f}")
        print(f"  Std Dev: {np.std(all_macro_f1s):.4f}")
    else:
        print("No macro-averaged F1-score data to process.")

    # Find and print the best replicate
    if replicates_data:
        best_f1_filename, best_f1_value = find_best_replicate(replicates_data, metric_key='macro_f1')
        print(f"\n--- Best Replicate (based on Macro-averaged F1-score) ---")
        if best_f1_filename:
            print(f"Filename: {best_f1_filename}")
            print(f"Macro-averaged F1-score: {best_f1_value:.4f}")
            # You could also print all metrics for the best run if desired
            # best_metrics = next(item['metrics'] for item in replicates_data if item['filename'] == best_f1_filename)
            # print(f"  All Metrics for Best Run: {best_metrics}")
        else:
            print("Could not determine the best replicate.")

if __name__ == "__main__":
    main()
