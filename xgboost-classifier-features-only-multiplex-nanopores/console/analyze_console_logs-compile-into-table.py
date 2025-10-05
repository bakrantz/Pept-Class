import os
import re
import numpy as np
import pandas as pd
from docx import Document

def extract_metrics_from_docx(docx_path):
    """
    Extracts overall accuracy, macro-averaged precision, recall, and F1-score
    from a .docx document containing a scikit-learn classification report.
    Returns a dictionary of metrics.
    """
    try:
        document = Document(docx_path)
        full_text = []
        for para in document.paragraphs:
            full_text.append(para.text)
        
        full_text_str = "\n".join(full_text)

        metrics = {}
        # Using a dictionary comprehension for cleaner extraction
        metric_patterns = {
            'accuracy': r"Overall Peptide Classification Accuracy:\s*(\d+\.\d+)",
            'macro_precision': r"Macro-averaged Precision:\s*(\d+\.\d+)",
            'macro_recall': r"Macro-averaged Recall:\s*(\d+\.\d+)",
            'macro_f1': r"Macro-averaged F1-score:\s*(\d+\.\d+)"
        }
        
        for key, pattern in metric_patterns.items():
            match = re.search(pattern, full_text_str)
            if match:
                metrics[key] = float(match.group(1))
            else:
                # If a metric is not found, default to None or a suitable value
                metrics[key] = None
        
        return metrics
    except Exception as e:
        print(f"Error processing {docx_path}: {e}")
        return {}


def main():
    """
    Main function to process all replicates, calculate summary statistics,
    and output a single CSV file.
    """
    # Define the base name of the files and the nanopore combinations to process
    file_base_name = "console-probabilistic-blending-ensemble"
    file_suffix = "-15ms"
    num_replicates = 5
    docx_directory = "."
    output_filename = "nanopore_ensemble_results.csv"

    # Define the nanopore combinations and their file name abbreviations
    # This dictionary makes the script easily extensible
    nanopore_combinations = {
        'WT': 'WT',
        'F427A': 'F427A',
        'F427Y': 'F427Y',
        'WT/F427A': 'WT_F427A',
        'WT/F427Y': 'WT_F427Y',
        'F427A/F427Y': 'F427A_F427Y',
        'WT/F427A/F427Y': 'WT_F427A_F427Y'
    }

    results = []

    print("--- Starting to process results for all nanopore combinations ---")

    # Iterate through each nanopore combination
    for display_name, file_abbrev in nanopore_combinations.items():
        print(f"\nProcessing combination: {display_name}")
        
        # Lists to hold the metrics for all replicates of the current combination
        combination_metrics = {
            'accuracy': [],
            'macro_precision': [],
            'macro_recall': [],
            'macro_f1': []
        }

        # Iterate through the replicates for the current combination
        for rpt_num in range(1, num_replicates + 1):
            filename = f"{file_base_name}-{file_abbrev}{file_suffix}-rpt{rpt_num}.docx"
            filepath = os.path.join(docx_directory, filename)

            if not os.path.exists(filepath):
                print(f"Warning: File not found - {filepath}. Skipping.")
                continue

            print(f"  - Processing {filename}...")
            metrics = extract_metrics_from_docx(filepath)

            if metrics:
                for key, value in metrics.items():
                    if value is not None:
                        combination_metrics[key].append(value)
        
        # Calculate mean and standard deviation for the current combination
        summary_row = {'Nanopores': display_name}
        for metric, values in combination_metrics.items():
            if values:
                summary_row[f'Mean {metric}'] = np.mean(values)
                summary_row[f'Std Dev {metric}'] = np.std(values)
            else:
                summary_row[f'Mean {metric}'] = None
                summary_row[f'Std Dev {metric}'] = None
        
        results.append(summary_row)

    # Create a DataFrame and save it to a CSV file
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_filename, index=False)
        print(f"\n--- Results saved to {output_filename} ---")
        print(results_df)
    else:
        print("\nNo data to save. Please check your file paths and names.")

if __name__ == "__main__":
    main()
