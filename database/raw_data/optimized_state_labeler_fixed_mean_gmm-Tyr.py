import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans

# --- Load ATF Function ---
def load_atf(filepath: str, header_row_index: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Loads data from an .atf file, preserving the multi-line header."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at '{filepath}'")

    with open(filepath, 'r') as f:
        all_lines = f.readlines()

    if len(all_lines) < header_row_index + 2:
        raise ValueError(f"File '{filepath}' is too short.")

    header_lines = [line.strip('\n') for line in all_lines[:header_row_index + 1]]

    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=header_row_index)
        df.columns = df.columns.str.strip().str.replace(' #', '').str.replace(' ', '_').str.replace('[()]', '', regex=True)

        required_cols = {"Time_s": "Time (s)", "Trace1_pA": "Trace #1 (pA)", "Trace1_mV": "Trace #1 (mV)"}
        
        times, current, voltage = None, None, None
        for cleaned_name, original_name in required_cols.items():
            if cleaned_name not in df.columns:
                raise KeyError(f"Expected column '{original_name}' not found.")
            if cleaned_name == "Time_s": times = df[cleaned_name].to_numpy()
            elif cleaned_name == "Trace1_pA": current = df[cleaned_name].to_numpy()
            elif cleaned_name == "Trace1_mV": voltage = df[cleaned_name].to_numpy()

        return times, current, voltage, header_lines

    except Exception as e:
        raise Exception(f"Error loading ATF '{filepath}': {e}")
    
# --- Constant Baseline Shift ---
def dynamic_baseline_correction(current_trace_data: np.ndarray, window_size: int = 50, threshold_std_dev: float = 3.0, n_clusters_for_baseline_detection: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Applies a constant baseline shift to anchor the open pore at 0 pA, bypassing moving-window drift."""
    if current_trace_data.size == 0:
        return np.array([]), np.array([])
    
    # First-pass clustering to find the baseline across the entire file
    kmeans_baseline = KMeans(n_clusters=n_clusters_for_baseline_detection, n_init='auto', random_state=42)
    kmeans_baseline.fit(current_trace_data.reshape(-1, 1))
    
    # FIXED: Use np.min to find the most negative peak (Open Pore) in trans-electrode recordings
    dominant_centroid = np.min(kmeans_baseline.cluster_centers_).flatten()[0]
    
    # Apply a constant shift to the entire trace (no moving window)
    corrected_current = current_trace_data - dominant_centroid
    
    # The drift vector is now just a flat, constant line for visualization
    drift_vector = np.full_like(current_trace_data, dominant_centroid)
    
    return corrected_current, drift_vector

# --- OPTIMIZED: Custom Fixed-Mean Gaussian Mixture Model ---
def identify_conductance_states(current_trace_data, initial_centroids, target_state_labels, filter_window=5, random_state=42):
    """
    Identifies conductance states using a Custom Fixed-Mean GMM.
    This forces the centroids to lock onto the expected physical fractions,
    but calculates the true variance of the noise so that wide peaks (State 1)
    do not get chopped off by narrow peaks (State 0).
    """
    if not isinstance(current_trace_data, np.ndarray) or current_trace_data.ndim > 1:
        current_trace_data = np.asarray(current_trace_data).flatten()
    
    n_states = len(initial_centroids)
    if len(target_state_labels) != n_states:
        raise ValueError("Length of target_state_labels must exactly match initial_centroids.")
    
    # 1. OPTIONAL DENOISING: Apply a median filter to isolate the true state peaks
    if filter_window > 1:
        smoothed_data = median_filter(current_trace_data, size=filter_window)
    else:
        smoothed_data = current_trace_data

    # 2. FIXED-MEAN GMM INITIALIZATION
    # Ensure means are perfectly sorted to match your target labels
    means = np.sort(np.array(initial_centroids))
    
    # Initial assignment via simple Euclidean distance (KMeans style)
    X = smoothed_data.reshape(-1, 1)
    distances = np.abs(X - means)
    labels = np.argmin(distances, axis=1)
    
    # 3. EXPECTATION-MAXIMIZATION (EM) LOOP (Means Locked)
    variances = np.zeros(n_states)
    weights = np.zeros(n_states)
    global_var = np.var(smoothed_data)
    
    for iteration in range(10): # 10 iterations is plenty for converging variances
        # M-Step: Update Variances and Weights (Means remain locked to physical reality)
        for i in range(n_states):
            mask = (labels == i)
            weights[i] = np.sum(mask) / len(smoothed_data)
            
            if weights[i] > 0.005: # If state has at least 0.5% of the data
                variances[i] = np.var(smoothed_data[mask])
            else:
                variances[i] = global_var * 0.1 # Fallback variance for rare states
                
        # Prevent zero-variance math crash
        variances = np.clip(variances, a_min=0.01, a_max=None)
        weights = np.clip(weights, a_min=1e-6, a_max=None)
        weights /= np.sum(weights) # Re-normalize probabilities
        
        # E-Step: Calculate Log-Likelihoods and Re-assign points
        log_probs = np.zeros((len(smoothed_data), n_states))
        for i in range(n_states):
            # log P(x | N(m, v)) + log P(component)
            log_pdf = -0.5 * np.log(2 * np.pi * variances[i]) - 0.5 * ((smoothed_data - means[i])**2 / variances[i])
            log_probs[:, i] = log_pdf + np.log(weights[i])
            
        new_labels = np.argmax(log_probs, axis=1)
        
        # Check for convergence
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

    # 4. EXPLICIT MAPPING
    label_mapping_array = np.zeros(n_states, dtype=int)
    for rank, old_label in enumerate(np.argsort(means)):
        label_mapping_array[old_label] = target_state_labels[rank]
        
    mapped_labels = label_mapping_array[labels]
    
    return means, mapped_labels

# --- Export Labeled CSV ---
def export_labeled_csv(filepath: str, times: np.ndarray, current: np.ndarray, labels: np.ndarray, output_dir: str):
    file_name_without_ext = os.path.splitext(os.path.basename(filepath))[0]
    csv_filename = f"{file_name_without_ext}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)

    df = pd.DataFrame({'Time': times, 'Current': current, 'State': labels})
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(csv_filepath, index=False)
    return csv_filepath

# --- Visualize Plots ---
def visualize_plots(times: np.ndarray, current: np.ndarray, drift_vector: np.ndarray, corrected_current: np.ndarray, centroids: np.ndarray, labels: np.ndarray, target_state_labels: list, title_suffix: str = ""):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(times, current, 'b-', label='Raw Current Data', alpha=0.7)
    ax1.plot(times, drift_vector, 'r-', label='Constant Baseline Reference', linewidth=2)
    ax1.set_title(f'Raw Current Data and Baseline Drift\n({title_suffix})')
    ax1.set_ylabel('Current (pA)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(times, corrected_current, 'g-', label='Corrected Current Data', alpha=0.7)
    ax2.set_title('Baseline-Corrected Current Trace')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (pA)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    counts, bin_edges = np.histogram(corrected_current, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.bar(bin_centers, counts, width=np.diff(bin_edges), color='skyblue', label='Current Distribution')
    plt.yscale('log')
    
    # Map the vertical lines to the explicit state labels
    for centroid, state_lbl in zip(centroids, target_state_labels):
        plt.axvline(centroid, color='red', linestyle='--', linewidth=1.5, 
                    label=f'State {state_lbl} Centroid: {centroid:.2f} pA')
    
    plt.title(f'Log-Scale Current Histogram (Fixed-Mean GMM)\n({title_suffix})')
    plt.xlabel('Current (pA)')
    plt.ylabel('Log(Count)')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# --- Batch Processor ---
def batch_processor(
    filepaths: list[str], 
    file_config_map: dict[str, dict],
    output_dir: str, 
    window_size: int = 4000, 
    threshold_std_dev: float = 1.0, 
    n_clusters_for_baseline_detection: int = 2, 
    filter_window: int = 5, 
    visualize_plots_bool: bool = True
):
    log_data = []
    os.makedirs(output_dir, exist_ok=True)
    
    # The physical fraction of max blockade for each state
    fractional_block = {3: 0.0, 2: 0.50, 1: 0.85, 0: 1.0}

    for filepath in filepaths:
        file_basename = os.path.basename(filepath)
        log_entry = {'atf_filename': file_basename, 'csv_filename': 'N/A', 'success': False, 'error_message': '', 'centroids': 'N/A'}

        try:
            config = file_config_map.get(file_basename)
            if config is None:
                raise ValueError(f"No configuration defined for file: {file_basename}")
                
            target_state_labels = config['state_labels']
            
            print(f"\n--- Processing {file_basename} ---")
            times, current, voltage, header = load_atf(filepath)
            
            # Baseline correct the data (Open pore becomes ~0 pA, Blocked states become positive)
            corrected_current, drift_vector = dynamic_baseline_correction(
                current, window_size=window_size, threshold_std_dev=threshold_std_dev,
                n_clusters_for_baseline_detection=n_clusters_for_baseline_detection
            )
            
            # --- AUTO-DETECT MAX AMPLITUDE FOR THIS SPECIFIC CHANNEL ---
            # Use median filter to strip high-frequency spikes so KMeans finds the true density peaks
            smoothed_for_peaks = median_filter(corrected_current, size=filter_window) if filter_window > 1 else corrected_current
            
            # Find the peaks in the baseline-corrected data
            kmeans_peaks = KMeans(n_clusters=len(target_state_labels), n_init='auto', random_state=42)
            kmeans_peaks.fit(smoothed_for_peaks.reshape(-1, 1))
            detected_peaks = np.sort(kmeans_peaks.cluster_centers_.flatten())
            
            # The highest current peak corresponds to the most blocked state the user asked for
            most_blocked_state_requested = min(target_state_labels) # e.g., 0 or 1
            highest_detected_peak = detected_peaks[-1]
            
            # Back-calculate the theoretical 100% max_amplitude based on the physics
            fractional_multiplier = fractional_block[most_blocked_state_requested]
            if fractional_multiplier > 0:
                auto_max_amplitude = highest_detected_peak / fractional_multiplier
            else:
                auto_max_amplitude = config.get('max_amplitude', 1.0) # Absolute fallback
                
            print(f"Auto-detected Channel Max Amplitude (100% Block): {auto_max_amplitude:.3f} pA")
            
            # --- GENERATE CENTROIDS ---
            unsorted_centroids = [fractional_block[state] * auto_max_amplitude for state in target_state_labels]
            initial_centroids_guess = np.sort(np.array(unsorted_centroids))
            
            # Ensure target_state_labels map perfectly to sorted centroids (Lowest current to Highest current)
            target_state_labels = sorted(target_state_labels, reverse=True) 

            print(f"GMM explicitly mapping {len(target_state_labels)} target states: {target_state_labels}")
            centroids, mapped_labels = identify_conductance_states(
                corrected_current, 
                initial_centroids=initial_centroids_guess, 
                target_state_labels=target_state_labels, 
                filter_window=filter_window
            )
            
            csv_filepath = export_labeled_csv(filepath, times, corrected_current, mapped_labels, output_dir)
            
            log_entry['success'] = True
            log_entry['csv_filename'] = os.path.basename(csv_filepath)
            log_entry['centroids'] = ', '.join([f"{c:.2f}" for c in centroids])
            print(f"Final GMM Centroids (pA): {centroids}")

            if visualize_plots_bool:
                visualize_plots(times, current, drift_vector, corrected_current, centroids, mapped_labels, target_state_labels, title_suffix=file_basename)

        except Exception as e:
            log_entry['success'] = False
            log_entry['error_message'] = str(e)
            print(f"Error processing {file_basename}: {e}")
        
        log_data.append(log_entry)

    log_df = pd.DataFrame(log_data)
    log_file_path = os.path.join(output_dir, 'processing_log.csv')
    log_df.to_csv(log_file_path, index=False)
    print(f"\nBatch processing complete. Log file saved to: {log_file_path}")

# --- Main block to run the batch processor ---
if __name__ == "__main__":
    # Define directories
    base_data_dir = "/Users/bakrantz/Desktop/guesthost_Tyr" 
    output_dir = "/Users/bakrantz/Desktop/guesthost_Tyr/processed_csvs" 

    # File paths string
    atf_filepaths_text_block = """
11n09001-guesthost_Tyr-20_mV-600_Hz.atf
11n09004-guesthost_Tyr-20_mV-600_Hz.atf
11n09004-guesthost_Tyr-40_mV-600_Hz.atf
11n16001-guesthost_Tyr-20_mV-600_Hz.atf
11n16002-guesthost_Tyr-30_mV-600_Hz-rpt_1.atf
11n16002-guesthost_Tyr-30_mV-600_Hz-rpt_2.atf
11n16002-guesthost_Tyr-75_mV-600_Hz.atf
11n16002-guesthost_Tyr-80_mV-600_Hz.atf
11n16003-guesthost_Tyr-15_mV-600_Hz.atf
11n16003-guesthost_Tyr-25_mV-600_Hz.atf
11n16003-guesthost_Tyr-30_mV-600_Hz.atf
11n16003-guesthost_Tyr-35_mV-600_Hz.atf
11n16003-guesthost_Tyr-40_mV-600_Hz.atf
11n16003-guesthost_Tyr-50_mV-600_Hz.atf
11n16003-guesthost_Tyr-60_mV-600_Hz.atf
11n16003-guesthost_Tyr-65_mV-600_Hz.atf
11n16003-guesthost_Tyr-75_mV-600_Hz.atf
11d02001-guesthost_Tyr-60_mV-600_Hz.atf
11d02001-guesthost_Tyr-80_mV-600_Hz.atf
11d02001-guesthost_Tyr-90_mV-600_Hz.atf
11d02003-guesthost_Tyr-15_mV-600_Hz.atf
11d02003-guesthost_Tyr-25_mV-600_Hz.atf
11d02003-guesthost_Tyr-30_mV-600_Hz.atf
11d02003-guesthost_Tyr-35_mV-600_Hz-rpt_1.atf
11d02003-guesthost_Tyr-35_mV-600_Hz-rpt_2.atf
11d02003-guesthost_Tyr-40_mV-600_Hz.atf
11d02003-guesthost_Tyr-50_mV-600_Hz.atf
11d02003-guesthost_Tyr-60_mV-600_Hz.atf
11d02003-guesthost_Tyr-65_mV-600_Hz-rpt_1.atf
11d02003-guesthost_Tyr-65_mV-600_Hz-rpt_2.atf
11d02003-guesthost_Tyr-90_mV-600_Hz.atf
11n09001-guesthost_Tyr-20_mV-400_Hz-downsampled.atf
11n09004-guesthost_Tyr-20_mV-400_Hz-downsampled.atf
11n09004-guesthost_Tyr-40_mV-400_Hz-downsampled.atf
11n16001-guesthost_Tyr-20_mV-400_Hz-downsampled.atf
11n16002-guesthost_Tyr-30_mV-400_Hz-downsampled-rpt_1.atf
11n16002-guesthost_Tyr-30_mV-400_Hz-downsampled-rpt_2.atf
11n16002-guesthost_Tyr-75_mV-400_Hz-downsampled.atf
11n16002-guesthost_Tyr-80_mV-400_Hz-downsampled.atf
11n16003-guesthost_Tyr-15_mV-400_Hz-downsampled.atf
11n16003-guesthost_Tyr-25_mV-400_Hz-downsampled.atf
11n16003-guesthost_Tyr-30_mV-400_Hz-downsampled.atf
11n16003-guesthost_Tyr-35_mV-400_Hz-downsampled.atf
11n16003-guesthost_Tyr-40_mV-400_Hz-downsampled.atf
11n16003-guesthost_Tyr-50_mV-400_Hz-downsampled.atf
11n16003-guesthost_Tyr-60_mV-400_Hz-downsampled.atf
11n16003-guesthost_Tyr-65_mV-400_Hz-downsampled.atf
11n16003-guesthost_Tyr-75_mV-400_Hz-downsampled.atf
11d02001-guesthost_Tyr-60_mV-400_Hz-downsampled.atf
11d02001-guesthost_Tyr-80_mV-400_Hz-downsampled.atf
11d02001-guesthost_Tyr-90_mV-400_Hz-downsampled.atf
11d02003-guesthost_Tyr-15_mV-400_Hz-downsampled.atf
11d02003-guesthost_Tyr-25_mV-400_Hz-downsampled.atf
11d02003-guesthost_Tyr-30_mV-400_Hz-downsampled.atf
11d02003-guesthost_Tyr-35_mV-400_Hz-downsampled-rpt_1.atf
11d02003-guesthost_Tyr-35_mV-400_Hz-downsampled-rpt_2.atf
11d02003-guesthost_Tyr-40_mV-400_Hz-downsampled.atf
11d02003-guesthost_Tyr-50_mV-400_Hz-downsampled.atf
11d02003-guesthost_Tyr-60_mV-400_Hz-downsampled.atf
11d02003-guesthost_Tyr-65_mV-400_Hz-downsampled-rpt_1.atf
11d02003-guesthost_Tyr-65_mV-400_Hz-downsampled-rpt_2.atf
11d02003-guesthost_Tyr-90_mV-400_Hz-downsampled.atf
11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf
11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf
11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf
11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_4.atf
11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_5.atf
11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_6.atf
11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_7.atf
11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf
11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf
11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf
11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_4.atf
11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_5.atf
11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_6.atf
11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_7.atf
11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf
11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf
11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_3.atf
11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_4.atf
11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_5.atf
11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_6.atf
11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_7.atf
11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf
11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf
11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_3.atf
11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_4.atf
11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_5.atf
11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_6.atf
11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_7.atf
"""

    atf_filepaths = [os.path.join(base_data_dir, line.strip()) for line in atf_filepaths_text_block.splitlines() if line.strip() and not line.startswith('#')]

    # Configurations simply define which states are present in the specific file.
    # The max_amplitude is retained purely as an absolute fallback in case auto-detection fails.
    
    config_10mv = {
        'max_amplitude': 0.665, 
        'state_labels': [3, 2, 0] 
    }
    config_15mv = {
        'max_amplitude': 1.07, 
        'state_labels': [3, 2, 1, 0] 
    }
    config_20mv = {
        'max_amplitude': 1.29, 
        'state_labels': [3, 2, 1, 0] 
    }        
    config_25mv = {
        'max_amplitude': 1.73, 
        'state_labels': [3, 2, 1, 0] 
    }   
    config_30mv = {
        'max_amplitude': 2.09,  
        'state_labels': [3, 2, 1, 0] 
    }
    config_35mv = {
        'max_amplitude': 2.44,  
        'state_labels': [3, 2, 1, 0] 
    }
    config_40mv = {
        'max_amplitude': 2.71,  
        'state_labels': [3, 2, 1, 0] 
    }
    config_50mv = {
        'max_amplitude': 3.35,  
        'state_labels': [3, 2, 1, 0] 
    }
    config_60mv = {
        'max_amplitude': 4.038,  
        'state_labels': [3, 2, 1, 0] 
    }
    config_65mv = {
        'max_amplitude': 4.384,  
        'state_labels': [3, 2, 1, 0] 
    }
    config_70mv = {
        'max_amplitude': 5.59,  
        'state_labels': [3, 2, 1, 0] 
    }
    config_75mv = {
        'max_amplitude': 4.977,  
        'state_labels': [3, 2, 1, 0] 
    }
    config_80mv = {
        'max_amplitude': 5.20,  
        'state_labels': [3, 2, 1, 0] 
    }
    config_90mv = {
        'max_amplitude': 6.54,  
        'state_labels': [3, 2, 1, 0] 
    }
    # Populate file_config_map based on voltage configuration
    file_config_map = {
        "11n09001-guesthost_Tyr-20_mV-600_Hz.atf": config_20mv,
        "11n09004-guesthost_Tyr-20_mV-600_Hz.atf": config_20mv,
        "11n09004-guesthost_Tyr-40_mV-600_Hz.atf": config_40mv,
        "11n16001-guesthost_Tyr-20_mV-600_Hz.atf": config_20mv,
        "11n16002-guesthost_Tyr-30_mV-600_Hz-rpt_1.atf": config_30mv,
        "11n16002-guesthost_Tyr-30_mV-600_Hz-rpt_2.atf": config_20mv,
        "11n16002-guesthost_Tyr-75_mV-600_Hz.atf": config_75mv,
        "11n16002-guesthost_Tyr-80_mV-600_Hz.atf": config_80mv,
        "11n16003-guesthost_Tyr-15_mV-600_Hz.atf": config_15mv,
        "11n16003-guesthost_Tyr-25_mV-600_Hz.atf": config_25mv,
        "11n16003-guesthost_Tyr-30_mV-600_Hz.atf": config_30mv,
        "11n16003-guesthost_Tyr-35_mV-600_Hz.atf": config_35mv,
        "11n16003-guesthost_Tyr-40_mV-600_Hz.atf": config_40mv,
        "11n16003-guesthost_Tyr-50_mV-600_Hz.atf": config_50mv,
        "11n16003-guesthost_Tyr-60_mV-600_Hz.atf": config_60mv,
        "11n16003-guesthost_Tyr-65_mV-600_Hz.atf": config_65mv,
        "11n16003-guesthost_Tyr-75_mV-600_Hz.atf": config_75mv,
        "11d02001-guesthost_Tyr-60_mV-600_Hz.atf": config_60mv,
        "11d02001-guesthost_Tyr-80_mV-600_Hz.atf": config_80mv,
        "11d02001-guesthost_Tyr-90_mV-600_Hz.atf": config_90mv,
        "11d02003-guesthost_Tyr-15_mV-600_Hz.atf": config_15mv,
        "11d02003-guesthost_Tyr-25_mV-600_Hz.atf": config_25mv,
        "11d02003-guesthost_Tyr-30_mV-600_Hz.atf": config_30mv,
        "11d02003-guesthost_Tyr-35_mV-600_Hz-rpt_1.atf": config_35mv,
        "11d02003-guesthost_Tyr-35_mV-600_Hz-rpt_2.atf": config_35mv,
        "11d02003-guesthost_Tyr-40_mV-600_Hz.atf": config_40mv,
        "11d02003-guesthost_Tyr-50_mV-600_Hz.atf": config_50mv,
        "11d02003-guesthost_Tyr-60_mV-600_Hz.atf": config_60mv,
        "11d02003-guesthost_Tyr-65_mV-600_Hz-rpt_1.atf": config_65mv,
        "11d02003-guesthost_Tyr-65_mV-600_Hz-rpt_2.atf": config_65mv,
        "11d02003-guesthost_Tyr-90_mV-600_Hz.atf": config_90mv,
        "11n09001-guesthost_Tyr-20_mV-400_Hz-downsampled.atf": config_20mv,
        "11n09004-guesthost_Tyr-20_mV-400_Hz-downsampled.atf": config_20mv,
        "11n09004-guesthost_Tyr-40_mV-400_Hz-downsampled.atf": config_40mv,
        "11n16001-guesthost_Tyr-20_mV-400_Hz-downsampled.atf": config_20mv,
        "11n16002-guesthost_Tyr-30_mV-400_Hz-downsampled-rpt_1.atf": config_30mv,
        "11n16002-guesthost_Tyr-30_mV-400_Hz-downsampled-rpt_2.atf": config_30mv,
        "11n16002-guesthost_Tyr-75_mV-400_Hz-downsampled.atf": config_75mv,
        "11n16002-guesthost_Tyr-80_mV-400_Hz-downsampled.atf": config_80mv,
        "11n16003-guesthost_Tyr-15_mV-400_Hz-downsampled.atf": config_15mv,
        "11n16003-guesthost_Tyr-25_mV-400_Hz-downsampled.atf": config_25mv,
        "11n16003-guesthost_Tyr-30_mV-400_Hz-downsampled.atf": config_30mv,
        "11n16003-guesthost_Tyr-35_mV-400_Hz-downsampled.atf": config_35mv,
        "11n16003-guesthost_Tyr-40_mV-400_Hz-downsampled.atf": config_40mv,
        "11n16003-guesthost_Tyr-50_mV-400_Hz-downsampled.atf": config_50mv,
        "11n16003-guesthost_Tyr-60_mV-400_Hz-downsampled.atf": config_60mv,
        "11n16003-guesthost_Tyr-65_mV-400_Hz-downsampled.atf": config_65mv,
        "11n16003-guesthost_Tyr-75_mV-400_Hz-downsampled.atf": config_75mv,
        "11d02001-guesthost_Tyr-60_mV-400_Hz-downsampled.atf": config_60mv,
        "11d02001-guesthost_Tyr-80_mV-400_Hz-downsampled.atf": config_80mv,
        "11d02001-guesthost_Tyr-90_mV-400_Hz-downsampled.atf": config_90mv,
        "11d02003-guesthost_Tyr-15_mV-400_Hz-downsampled.atf": config_15mv,
        "11d02003-guesthost_Tyr-25_mV-400_Hz-downsampled.atf": config_25mv,
        "11d02003-guesthost_Tyr-30_mV-400_Hz-downsampled.atf": config_30mv,
        "11d02003-guesthost_Tyr-35_mV-400_Hz-downsampled-rpt_1.atf": config_35mv,
        "11d02003-guesthost_Tyr-35_mV-400_Hz-downsampled-rpt_2.atf": config_35mv,
        "11d02003-guesthost_Tyr-40_mV-400_Hz-downsampled.atf": config_40mv,
        "11d02003-guesthost_Tyr-50_mV-400_Hz-downsampled.atf": config_50mv,
        "11d02003-guesthost_Tyr-60_mV-400_Hz-downsampled.atf": config_60mv,
        "11d02003-guesthost_Tyr-65_mV-400_Hz-downsampled-rpt_1.atf": config_65mv,
        "11d02003-guesthost_Tyr-65_mV-400_Hz-downsampled-rpt_2.atf": config_65mv,
        "11d02003-guesthost_Tyr-90_mV-400_Hz-downsampled.atf": config_90mv,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_4.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_5.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_6.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_7.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_4.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_5.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_6.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_7.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_3.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_4.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_5.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_6.atf": config_70mv,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_7.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_3.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_4.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_5.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_6.atf": config_70mv,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_7.atf": config_70mv
    }

    if atf_filepaths:
        batch_processor(
            filepaths=atf_filepaths,
            file_config_map=file_config_map,
            output_dir=output_dir,
            window_size=4000,
            threshold_std_dev=1.0,
            n_clusters_for_baseline_detection=3,
            filter_window=5, 
            visualize_plots_bool=True
        )
