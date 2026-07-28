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
    
    # 1. CALCULATE PERCENTILES TO DEFEAT ZINGER SPIKES
    # Drop the extreme top and bottom 0.1% of data points to find the true biological range
    raw_min, raw_max = np.percentile(current, [0.1, 99.9])
    raw_margin = (raw_max - raw_min) * 0.10 # Add 10% visual padding
    
    corr_min, corr_max = np.percentile(corrected_current, [0.1, 99.9])
    corr_margin = (corr_max - corr_min) * 0.10

    # 2. PLOT THE TIME TRACES
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(times, current, 'b-', label='Raw Current Data', alpha=0.7)
    ax1.plot(times, drift_vector, 'r-', label='Constant Baseline Reference', linewidth=2)
    ax1.set_ylim(raw_min - raw_margin, raw_max + raw_margin) # Force Y-axis to ignore spikes
    ax1.set_title(f'Raw Current Data and Baseline Drift\n({title_suffix})')
    ax1.set_ylabel('Current (pA)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(times, corrected_current, 'g-', label='Corrected Current Data', alpha=0.7)
    ax2.set_ylim(corr_min - corr_margin, corr_max + corr_margin) # Force Y-axis to ignore spikes
    ax2.set_title('Baseline-Corrected Current Trace')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (pA)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    # 3. PLOT THE HISTOGRAM
    plt.figure(figsize=(12, 6))
    
    # Force the histogram to ONLY bin the data within our safe percentile range
    hist_range = (corr_min - corr_margin, corr_max + corr_margin)
    counts, bin_edges = np.histogram(corrected_current, bins=200, range=hist_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.bar(bin_centers, counts, width=np.diff(bin_edges), color='skyblue', label='Current Distribution')
    plt.yscale('log')
    plt.xlim(hist_range) # Force X-axis to zoom in on the peaks
    
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

    for filepath in filepaths:
        file_basename = os.path.basename(filepath)
        log_entry = {'atf_filename': file_basename, 'csv_filename': 'N/A', 'success': False, 'error_message': '', 'centroids': 'N/A'}

        try:
            config = file_config_map.get(file_basename)
            if config is None:
                raise ValueError(f"No configuration defined for file: {file_basename}")
                
            target_state_labels = config['state_labels']
            
            # NEW: Pull the specific fractional block for this peptide, or fallback to the aromatic default
            fractional_block = config.get('fractional_block', {3: 0.0, 2: 0.50, 1: 0.85, 0: 1.0})
            
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

if __name__ == "__main__":
    # Define directories
    base_data_dir = "/Users/bakrantz/Desktop/guesthost_Phe" 
    output_dir = "/Users/bakrantz/Desktop/guesthost_Phe" 

    # File paths string
    atf_filepaths_text_block = """
11803004-guesthost_Phe-15_mV-400_Hz-rpt_1.atf
11803004-guesthost_Phe-15_mV-400_Hz-rpt_2.atf
11803004-guesthost_Phe-70_mV-400_Hz.atf
11803003-guesthost_Phe-15_mV-400_Hz.atf
11803003-guesthost_Phe-20_mV-400_Hz.atf
11803003-guesthost_Phe-90_mV-400_Hz.atf
11803003-guesthost_Phe-27_mV-400_Hz.atf
11803003-guesthost_Phe-22_mV-400_Hz.atf
11803003-guesthost_Phe-10_mV-400_Hz.atf
11803002-guesthost_Phe-20_mV-400_Hz.atf
11803002-guesthost_Phe-90_mV-400_Hz-rpt_1.atf
11803002-guesthost_Phe-90_mV-400_Hz-rpt_2.atf
11803002-guesthost_Phe-90_mV-400_Hz-rpt_3.atf
11803002-guesthost_Phe-70_mV-400_Hz-rpt_1.atf
11803002-guesthost_Phe-70_mV-400_Hz-rpt_2.atf
11620003-guesthost_Phe-22_mV-400_Hz.atf
11620003-guesthost_Phe-25_mV-400_Hz.atf
11620003-guesthost_Phe-80_mV-400_Hz.atf
11620002-guesthost_Phe-20_mV-400_Hz.atf
11620002-guesthost_Phe-80_mV-400_Hz.atf
11620005-guesthost_Phe-60_mV-400_Hz.atf
11620005-guesthost_Phe-40_mV-400_Hz-rpt_1.atf
11620005-guesthost_Phe-40_mV-400_Hz-rpt_2.atf
11620005-guesthost_Phe-35_mV-400_Hz-rpt_1.atf
11620005-guesthost_Phe-35_mV-400_Hz-rpt_2.atf
11620005-guesthost_Phe-25_mV-400_Hz-rpt_1.atf
11620005-guesthost_Phe-25_mV-400_Hz-rpt_2.atf
11620005-guesthost_Phe-25_mV-400_Hz-rpt_3.atf
11620005-guesthost_Phe-50_mV-400_Hz.atf
11620005-guesthost_Phe-27_mV-400_Hz.atf
11620005-guesthost_Phe-80_mV-400_Hz.atf
11620005-guesthost_Phe-30_mV-400_Hz.atf
11622001-guesthost_Phe-22_mV-400_Hz.atf
11622001-guesthost_Phe-25_mV-400_Hz.atf
11622001-guesthost_Phe-20_mV-400_Hz.atf
11622001-guesthost_Phe-80_mV-400_Hz.atf
11622001-guesthost_Phe-30_mV-400_Hz.atf
11622000-guesthost_Phe-70_mV-400_Hz.atf
11620000-guesthost_Phe-60_mV-400_Hz.atf
11620000-guesthost_Phe-22_mV-400_Hz.atf
11620000-guesthost_Phe-35_mV-400_Hz.atf
11620000-guesthost_Phe-70_mV-400_Hz.atf
11620000-guesthost_Phe-20_mV-400_Hz-rpt_1.atf
11620000-guesthost_Phe-20_mV-400_Hz-rpt_2.atf
11620000-guesthost_Phe-25_mV-400_Hz.atf
11620000-guesthost_Phe-40_mV-400_Hz.atf
11620000-guesthost_Phe-50_mV-400_Hz.atf
11620000-guesthost_Phe-27_mV-400_Hz.atf
11620000-guesthost_Phe-30_mV-400_Hz.atf
11620001-guesthost_Phe-70_mV-400_Hz-rpt_1.atf
11620001-guesthost_Phe-70_mV-400_Hz-rpt_2.atf
11622003-guesthost_Phe-22_mV-400_Hz.atf
11622003-guesthost_Phe-70_mV-400_Hz.atf
11622003-guesthost_Phe-40_mV-400_Hz.atf
11622003-guesthost_Phe-80_mV-400_Hz.atf
11622004-guesthost_Phe-15_mV-400_Hz-rpt_1.atf
11622004-guesthost_Phe-15_mV-400_Hz-rpt_2.atf
11622004-guesthost_Phe-35_mV-400_Hz.atf
11622004-guesthost_Phe-27_mV-400_Hz.atf
"""

    atf_filepaths = [os.path.join(base_data_dir, line.strip()) for line in atf_filepaths_text_block.splitlines() if line.strip() and not line.startswith('#')]

    # The physical fraction of max blockade for Alanine based on empirical histograms other fractions are given if needed
    ala_fractions = {3: 0.0, 2: 0.35, 1: 0.8, 0: 1.0}
    leu_fractions = {3: 0.0, 2: 0.5, 1: 0.80, 0: 1.0}
    aromatic_fractions = {3: 0.0, 2: 0.5, 1: 0.85, 0: 1.0}
    
    # Configurations simply define which states are present in the specific file.
    # The max_amplitude is retained purely as an absolute fallback in case auto-detection fails.
    # Zero-Intercept Fit Results for Open Pore Currents
    # Conductance (Slope): 0.0682 pA/mV (nS); Fit Quality (R²): 0.9777

    config_10mv = {
        'max_amplitude': 0.682, 
        'state_labels': [3, 2, 0],
        'fractional_block': aromatic_fractions
    }
    config_15mv = {
        'max_amplitude': 1.023, 
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_20mv = {
        'max_amplitude': 1.363, 
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_22mv = {
        'max_amplitude': 1.500, 
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_25mv = {
        'max_amplitude': 1.704, 
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_27mv = {
        'max_amplitude': 1.841, 
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_30mv = {
        'max_amplitude': 2.045,  
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_35mv = {
        'max_amplitude': 2.386,  
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_40mv = {
        'max_amplitude': 2.727,  
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_50mv = {
        'max_amplitude': 3.408,  
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_60mv = {
        'max_amplitude': 4.090,  
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_65mv = {
        'max_amplitude': 4.431,  
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_70mv = {
        'max_amplitude': 4.772,  
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_75mv = {
        'max_amplitude': 5.113,  
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_80mv = {
        'max_amplitude': 5.453,  
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }
    config_90mv = {
        'max_amplitude': 6.135,  
        'state_labels': [3, 2, 1, 0],
        'fractional_block': aromatic_fractions
    }

    # --- DYNAMICALLY BUILD file_config_map ---
    # 1. Map the integer voltage directly to your defined config dictionaries
    voltage_configs = {
        10: config_10mv, 15: config_15mv, 20: config_20mv, 22: config_22mv,
        25: config_25mv, 27: config_27mv, 30: config_30mv, 35: config_35mv,
        40: config_40mv, 50: config_50mv, 60: config_60mv, 65: config_65mv,
        70: config_70mv, 75: config_75mv, 80: config_80mv, 90: config_90mv
    }
    
    import re
    file_config_map = {}
    
    # 2. Iterate through the filenames and extract the voltage using regex
    for filepath in atf_filepaths:
        filename = os.path.basename(filepath)
        
        # Regex looks for a dash, any number of digits, and "_mV" (e.g., "-70_mV")
        match = re.search(r'-(\d+)_mV', filename)
        
        if match:
            volt_val = int(match.group(1))
            if volt_val in voltage_configs:
                file_config_map[filename] = voltage_configs[volt_val]
            else:
                print(f"⚠️ Warning: No config defined for {volt_val} mV ({filename})")
        else:
            print(f"⚠️ Warning: Could not parse voltage from {filename}")

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
