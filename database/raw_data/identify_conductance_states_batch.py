import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Your provided load_atf function (copied here for completeness) ---
def load_atf(filepath: str, header_row_index: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Loads data from an .atf file, preserving the multi-line header and
    extracting time, current, and voltage as NumPy arrays.
    
    Args:
        filepath (str): The full path to the .atf file.
        header_row_index (int, optional): The 0-indexed row number where the column
                                            names are located. Defaults to 9 (10th line).
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]: A tuple containing:
            - times (np.ndarray): NumPy array of time values (in seconds).
            - current (np.ndarray): NumPy array of current values (in pA).
            - voltage (np.ndarray): NumPy array of voltage values (in mV).
            - header_lines (list[str]): A list of strings, where each string is
                                        a line from the ATF header, including the
                                        column names line.
    
    Raises:
        FileNotFoundError: If the specified filepath does not exist.
        KeyError: If expected column names are not found in the file.
        ValueError: If data cannot be parsed into numeric types.
        Exception: For other potential errors during file reading.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at '{filepath}'")

    all_lines = []
    with open(filepath, 'r') as f:
        all_lines = f.readlines()

    if len(all_lines) < header_row_index + 2: # At least header line + 1 data row
        raise ValueError(f"File '{filepath}' is too short to contain header and data as expected.")

    # Extract header lines (from ATF 1.0 to column names line, inclusive)
    header_lines = [line.strip('\n') for line in all_lines[:header_row_index + 1]]

    # Read data using pandas, skipping header lines
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=header_row_index)

        # Clean column names similar to previous functions
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(' #', '')
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('[()]', '', regex=True)

        required_cols = {
            "Time_s": "Time (s)",
            "Trace1_pA": "Trace #1 (pA)",
            "Trace1_mV": "Trace #1 (mV)"
        }
        
        # Check if expected cleaned columns exist and retrieve them
        times = None
        current = None
        voltage = None

        for cleaned_name, original_name in required_cols.items():
            if cleaned_name not in df.columns:
                raise KeyError(f"Expected column '{original_name}' (cleaned to '{cleaned_name}') not found in file '{filepath}'. Available: {df.columns.tolist()}")
            
            if cleaned_name == "Time_s":
                times = df[cleaned_name].to_numpy()
            elif cleaned_name == "Trace1_pA":
                current = df[cleaned_name].to_numpy()
            elif cleaned_name == "Trace1_mV":
                voltage = df[cleaned_name].to_numpy()

        # Ensure all arrays are extracted
        if times is None or current is None or voltage is None:
            raise ValueError(f"Could not extract all required data columns from '{filepath}'.")

        # Basic type conversion check (pandas usually handles this, but good to be explicit)
        if not (np.issubdtype(times.dtype, np.number) and 
                        np.issubdtype(current.dtype, np.number) and 
                        np.issubdtype(voltage.dtype, np.number)):
            raise ValueError(f"Data in '{filepath}' could not be entirely converted to numeric types.")

        return times, current, voltage, header_lines

    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{filepath}' is empty or has no data after headers.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing data section of '{filepath}': {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading ATF '{filepath}': {e}")
    
# --- New function to handle dynamic baseline correction ---
def dynamic_baseline_correction(current_trace_data: np.ndarray, window_size: int = 50, threshold_std_dev: float = 3.0, n_clusters_for_baseline_detection: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs dynamic baseline correction by:
    1. Identifying the dominant state (the baseline) using a first-pass KMeans clustering.
       It will select the lowest current centroid from this initial clustering as the baseline.
    2. Finding a moving average of the identified baseline points.
    3. Subtracting this moving average drift from the entire signal.

    Args:
        current_trace_data (np.ndarray): 1D array of raw current values (in pA).
        window_size (int, optional): The number of data points for the moving average window.
                                     Defaults to 50.
        threshold_std_dev (float, optional): The number of standard deviations from the
                                            dominant state centroid to identify baseline points.
                                            Defaults to 3.0.
        n_clusters_for_baseline_detection (int, optional): Number of clusters KMeans
                                                            will search for in the first pass
                                                            to identify the baseline. The lowest
                                                            current centroid among these will be
                                                            chosen as the baseline. Defaults to 1.
                                            
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - corrected_current (np.ndarray): The current trace with baseline drift removed.
            - drift_vector (np.ndarray): The computed drift curve.
    """
    if current_trace_data.size == 0:
        return np.array([]), np.array([])
    
    # First-pass clustering to find the dominant state(s)
    # Using n_init='auto' for robustness.
    kmeans_baseline = KMeans(n_clusters=n_clusters_for_baseline_detection, n_init='auto', random_state=42)
    kmeans_baseline.fit(current_trace_data.reshape(-1, 1))
    
    # Determine the baseline centroid by selecting the lowest current centroid
    # from the initial KMeans clustering.
    dominant_centroid = np.min(kmeans_baseline.cluster_centers_).flatten()[0]
    
    # Calculate the standard deviation of the entire trace
    std_dev = np.std(current_trace_data)
    
    # Create a boolean mask to identify points within the threshold of the dominant centroid
    is_baseline = np.abs(current_trace_data - dominant_centroid) < threshold_std_dev * std_dev
    
    # Compute a moving average of the baseline points
    drift_points = []
    time_points = []
    
    # Iterate through the data in windows
    for i in range(0, len(current_trace_data), window_size):
        window_end = i + window_size
        window_slice = is_baseline[i:window_end]
        
        # Check if there are any baseline points in the current window
        if np.any(window_slice):
            baseline_currents_in_window = current_trace_data[i:window_end][window_slice]
            drift_points.append(np.mean(baseline_currents_in_window))
            time_points.append(i + window_size / 2) # Use the center of the window as the time point
        elif len(drift_points) > 0:
            # If no baseline points, carry forward the last good value
            drift_points.append(drift_points[-1])
            time_points.append(i + window_size / 2)
        else:
            # Handle the case where no baseline points are found at the beginning of the trace
            drift_points.append(dominant_centroid)
            time_points.append(i + window_size / 2)

    # Convert lists to numpy arrays for interpolation
    drift_points = np.array(drift_points)
    time_points = np.array(time_points)

    # Use linear interpolation to create a drift vector of the same length as the original data
    x_original = np.arange(len(current_trace_data))
    drift_interpolator = interp1d(time_points, drift_points, kind='linear', fill_value='extrapolate')
    drift_vector = drift_interpolator(x_original)
    
    # Subtract the drift vector from the original signal to correct the baseline
    corrected_current = current_trace_data - drift_vector
    
    return corrected_current, drift_vector

# --- The identify_conductance_states function (updated to use a more direct classification method) ---
def identify_conductance_states(current_trace_data: np.ndarray, n_states: int, initial_centroids: np.ndarray | None = None, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies conductance states in a current trace by directly classifying
    data points based on user-provided centroids. This is a more 'heavy-handed'
    method to ensure the algorithm finds a specific set of states, especially
    when dealing with imbalanced populations.
    
    Args:
        current_trace_data (np.ndarray): 1D array of current values (in pA).
        n_states (int): The number of expected conductance states.
        initial_centroids (np.ndarray): 1D array of initial centroid values.
                                        Must have a length equal to n_states.
                                        This parameter is now REQUIRED.
        random_state (int): Seed for reproducibility of KMeans. Set to None for random initialization.
    
    Returns:
        tuple: (centroids, labels)
            centroids (np.ndarray): 1D array of the mean current for each state, sorted.
            labels (np.ndarray): 1D array of state labels (0 to n_states-1) for each data point
                                 in current_trace_data, mapped to sorted centroids.
    """
    if not isinstance(current_trace_data, np.ndarray) or current_trace_data.ndim > 1:
        current_trace_data = np.asarray(current_trace_data).flatten()
    
    if initial_centroids is None or len(initial_centroids) != n_states:
        raise ValueError(f"initial_centroids must be a 1D array of length equal to n_states.")
    
    # Reshape for distance calculations
    X = current_trace_data.reshape(-1, 1)
    
    # Calculate the distance from each data point to each of the initial centroids
    distances = np.abs(X - initial_centroids)
    
    # Assign each data point to the state with the minimum distance
    labels = np.argmin(distances, axis=1)
    
    # Calculate the final centroids as the mean of the data points for each label
    final_centroids = np.zeros(n_states)
    for i in range(n_states):
        points_in_state = current_trace_data[labels == i]
        if points_in_state.size > 0:
            final_centroids[i] = np.mean(points_in_state)
        else:
            final_centroids[i] = initial_centroids[i] # If no points, keep initial guess
            
    # Sort the centroids and re-map labels for consistent ordering
    sorted_indices = np.argsort(final_centroids)
    sorted_centroids = final_centroids[sorted_indices]
    
    # Create a mapping from old label to new sorted label
    old_to_new_label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
    mapped_labels = np.array([old_to_new_label_map[label] for label in labels])
    
    return sorted_centroids, mapped_labels


# --- NEW FUNCTION: Export labeled data to a CSV file ---
def export_labeled_csv(filepath: str, times: np.ndarray, current: np.ndarray, labels: np.ndarray, output_dir: str):
    """
    Exports the time, corrected current, and state labels to a CSV file.

    Args:
        filepath (str): The original full path to the .atf file.
        times (np.ndarray): 1D array of time values.
        current (np.ndarray): 1D array of corrected current values.
        labels (np.ndarray): 1D array of state labels.
        output_dir (str): The directory where the CSV file will be saved.
    """
    # Create the new CSV filename
    file_name_without_ext = os.path.splitext(os.path.basename(filepath))[0]
    csv_filename = f"{file_name_without_ext}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)

    # Create a DataFrame from the data with capitalized headers
    df = pd.DataFrame({
        'Time': times,
        'Current': current,
        'State': labels
    })

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Export to CSV
    df.to_csv(csv_filepath, index=False)
    return csv_filepath

# --- NEW FUNCTION: Encapsulate all plotting logic ---
def visualize_plots(times: np.ndarray, current: np.ndarray, drift_vector: np.ndarray, corrected_current: np.ndarray, centroids: np.ndarray, labels: np.ndarray, title_suffix: str = ""):
    """
    Creates and displays the two plots for the data:
    1. Raw data with baseline drift.
    2. Log-scale histogram of corrected data with centroids.

    Args:
        times (np.ndarray): Array of time values.
        current (np.ndarray): Array of raw current values.
        drift_vector (np.ndarray): Array of the computed drift curve.
        corrected_current (np.ndarray): Array of baseline-corrected current values.
        centroids (np.ndarray): Array of identified centroids.
        labels (np.ndarray): Array of state labels.
        title_suffix (str): A suffix to add to the plot titles, e.g., the filename.
    """
    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top subplot: Raw Current Data with Drift Curve
    ax1.plot(times, current, 'b-', label='Raw Current Data', alpha=0.7)
    ax1.plot(times, drift_vector, 'r-', label='Computed Drift Curve', linewidth=2)
    ax1.set_title(f'Raw Current Data and Computed Baseline Drift\n({title_suffix})')
    ax1.set_ylabel('Current (pA)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Bottom subplot: Baseline-Corrected Current Data
    ax2.plot(times, corrected_current, 'g-', label='Corrected Current Data', alpha=0.7)
    ax2.set_title('Drift-Corrected Current Trace')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (pA)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show()

    # Create a second figure for the log-scale histogram
    plt.figure(figsize=(12, 6))
    
    # Create a histogram of the corrected current data
    counts, bin_edges = np.histogram(corrected_current, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plotting on a log scale
    plt.bar(bin_centers, counts, width=np.diff(bin_edges), color='skyblue', label='Current Distribution')
    plt.yscale('log')
    
    # Plot vertical lines at each identified centroid
    for i, centroid in enumerate(centroids):
        plt.axvline(centroid, color='red', linestyle='--', linewidth=1.5, 
                    label=f'State {i} Centroid: {centroid:.2f} pA')
    
    plt.title(f'Log-Scale Current Histogram with {len(centroids)} Conductance States\n({title_suffix})')
    plt.xlabel('Current (pA)')
    plt.ylabel('Log(Count)')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# --- NEW FUNCTION: Batch processor for multiple files ---
def batch_processor(
    filepaths: list[str], 
    initial_centroids_map: dict[str, np.ndarray], # Changed to a dictionary
    output_dir: str, # New keyword argument for output directory
    window_size: int = 4000, 
    threshold_std_dev: float = 1.0, 
    n_clusters_for_baseline_detection: int = 2, # Default to 2 as discussed
    visualize_plots_bool: bool = True
):
    """
    Processes a batch of .atf files, identifying conductance states and saving results.

    Args:
        filepaths (list[str]): A list of full paths to the .atf files to process.
        initial_centroids_map (dict[str, np.ndarray]): A dictionary mapping
                                                       filename basenames to their
                                                       corresponding initial centroid guesses.
        output_dir (str): The directory where the resulting labeled .csv files and log will be saved.
        window_size (int, optional): Window size for baseline correction. Defaults to 4000.
        threshold_std_dev (float, optional): Threshold for baseline correction. Defaults to 1.0.
        n_clusters_for_baseline_detection (int, optional): Number of clusters KMeans
                                                            will search for in the first pass
                                                            to identify the baseline. The lowest
                                                            current centroid among these will be
                                                            chosen as the baseline. Defaults to 2.
        visualize_plots_bool (bool, optional): Whether to display plots for each file. Defaults to True.
    """
    log_data = []

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for filepath in filepaths:
        file_basename = os.path.basename(filepath)
        log_entry = {
            'atf_filename': file_basename,
            'csv_filename': 'N/A',
            'success': False,
            'error_message': '',
            'centroids': 'N/A'
        }

        try:
            # Get initial centroids for the current file
            initial_centroids_guess_for_file = initial_centroids_map.get(file_basename)
            if initial_centroids_guess_for_file is None:
                raise ValueError(f"No initial centroids defined for file: {file_basename}. Please add it to initial_centroids_map.")

            print(f"\n--- Processing {file_basename} ---")
            times, current, voltage, header = load_atf(filepath)
            print(f"Successfully loaded {len(current)} data points.")
            
            # Pass the new baseline detection parameter
            corrected_current, drift_vector = dynamic_baseline_correction(
                current, 
                window_size=window_size, 
                threshold_std_dev=threshold_std_dev,
                n_clusters_for_baseline_detection=n_clusters_for_baseline_detection # Pass this through
            )
            
            n_states = len(initial_centroids_guess_for_file)
            print(f"Identifying {n_states} conductance states on corrected data...")
            centroids, labels = identify_conductance_states(corrected_current, n_states, initial_centroids=initial_centroids_guess_for_file)

            # --- MODIFICATION TO REVERSE LABELS ---
            # This reverses the labels to match the convention: 0=blocked, 3=open
            reversed_labels = (n_states - 1) - labels
            
            # Export the labeled data to CSV using the new reversed labels
            # Pass the output_dir to the export_labeled_csv function
            csv_filepath = export_labeled_csv(filepath, times, corrected_current, reversed_labels, output_dir)
            
            # Update log entry for success
            log_entry['success'] = True
            log_entry['csv_filename'] = os.path.basename(csv_filepath)
            log_entry['centroids'] = ', '.join([f"{c:.2f}" for c in centroids])
            print(f"Identified Centroids (pA): {centroids}")
            print(f"Labeled data exported to: {os.path.basename(csv_filepath)}")

            # Display plots if requested
            if visualize_plots_bool:
                visualize_plots(times, current, drift_vector, corrected_current, centroids, reversed_labels, title_suffix=file_basename)

        except Exception as e:
            log_entry['success'] = False
            log_entry['error_message'] = str(e)
            print(f"Error processing {file_basename}: {e}")
        
        log_data.append(log_entry)

    # Export the processing log to the specified output_dir
    log_file_path = os.path.join(output_dir, 'processing_log.csv')
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(log_file_path, index=False)
    print(f"\nBatch processing complete. Log file saved to: {log_file_path}")

# --- Main block to run the batch processor ---
if __name__ == "__main__":
    # Define the output directory for the batch
    # Make sure this directory exists or will be created by os.makedirs
    output_dir = "./PA/guesthost_Tyr/" # Changed to a dedicated processed_data folder

    # File list as a text block so you can easily paste from spreadsheet without having to add quotes and commas
    atf_filepaths_text_block = """
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
11n09003-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf
11n09003-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf
11n09004-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf
11n09004-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf
11n09004-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf
11n16003-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf
11n16003-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf
11801001-guesthost_Tyr-70mV-400_Hz-rpt_1.atf
11801001-guesthost_Tyr-70mV-400_Hz-rpt_2.atf
11801001-guesthost_Tyr-70mV-400_Hz-rpt_3.atf
11802000-guesthost_Tyr-70_mV-400_Hz.atf
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
11n09003-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf
11n09003-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf
11n09004-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf
11n09004-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf
11n09004-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_3.atf
11n16003-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf
11n16003-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf
"""
    # Construct full paths relative to the script's directory
    # You will need to adjust 'base_data_dir' to point to the parent directory
    base_data_dir = "/Users/bakrantz/Documents/python/database/raw_data/PA/guesthost_Tyr/" # Base data directory
    atf_filepaths = [os.path.join(base_data_dir, line.strip()) for line in atf_filepaths_text_block.splitlines() if line.strip()]

    # Define different types of centroid guesses for conductance states
    # Some channel sizes vary due to electronic artifacts, alternate channel oligomerization states, or membrane thickness differences

    # Type 1
    centroids_type1 = np.array([0, 2.0, 3.9, 4.7]) 
    
    # Type 2
    centroids_type2 = np.array([0, 2.819, 4.343, 5.389])

    # Type 3
    centroids_type3 = np.array([0, 2.28, 3.7995, 4.635]) 

    # Define more types if necessary
    
    # Map filenames to their corresponding initial centroid guesses
    # You'll need to populate this dictionary based on your knowledge of which files belong to which type.
    initial_centroids_map = {
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_4.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_5.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_6.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_7.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_4.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_5.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_6.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_7.atf": centroids_type1,
        "11n09003-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf": centroids_type1,
        "11n09003-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf": centroids_type1,
        "11n09004-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf": centroids_type1,
        "11n09004-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf": centroids_type1,
        "11n09004-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf": centroids_type1,
        "11n16003-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf": centroids_type1,
        "11n16003-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf": centroids_type1,
        "11801001-guesthost_Tyr-70mV-400_Hz-rpt_1.atf": centroids_type1,
        "11801001-guesthost_Tyr-70mV-400_Hz-rpt_2.atf": centroids_type1,
        "11801001-guesthost_Tyr-70mV-400_Hz-rpt_3.atf": centroids_type1,
        "11802000-guesthost_Tyr-70_mV-400_Hz.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_3.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_4.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_5.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_6.atf": centroids_type1,
        "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_7.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_3.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_4.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_5.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_6.atf": centroids_type1,
        "11n09002-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_7.atf": centroids_type1,
        "11n09003-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf": centroids_type1,
        "11n09003-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf": centroids_type1,
        "11n09004-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf": centroids_type1,
        "11n09004-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf": centroids_type1,
        "11n09004-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_3.atf": centroids_type1,
        "11n16003-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf": centroids_type1,
        "11n16003-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_2.atf": centroids_type1,
    }

    # Run the batch processor
    batch_processor(
        filepaths=atf_filepaths,
        initial_centroids_map=initial_centroids_map, # Pass the map
        output_dir=output_dir,
        window_size=4000,
        threshold_std_dev=1.0,
        n_clusters_for_baseline_detection=3,
        visualize_plots_bool=True
    )
