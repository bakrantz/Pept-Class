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
def dynamic_baseline_correction(current_trace_data: np.ndarray, window_size: int = 50, threshold_std_dev: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs dynamic baseline correction by:
    1. Identifying the dominant state (the baseline) using a first-pass KMeans clustering.
    2. Finding a moving average of the identified baseline points.
    3. Subtracting this moving average drift from the entire signal.

    Args:
        current_trace_data (np.ndarray): 1D array of raw current values (in pA).
        window_size (int, optional): The number of data points for the moving average window.
                                     Defaults to 50.
        threshold_std_dev (float, optional): The number of standard deviations from the
                                            dominant state centroid to identify baseline points.
                                            Defaults to 3.0.
                                            
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - corrected_current (np.ndarray): The current trace with baseline drift removed.
            - drift_vector (np.ndarray): The computed drift curve.
    """
    if current_trace_data.size == 0:
        return np.array([]), np.array([])
    
    # First-pass clustering to find the dominant state's centroid
    # Using n_init='auto' for robustness.
    kmeans = KMeans(n_clusters=1, n_init='auto', random_state=42)
    kmeans.fit(current_trace_data.reshape(-1, 1))
    dominant_centroid = kmeans.cluster_centers_.flatten()[0]
    
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

# --- The identify_conductance_states function (updated to accept initial centroids) ---
def identify_conductance_states(current_trace_data: np.ndarray, n_states: int, initial_centroids: np.ndarray | None = None, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies conductance states in a current trace using KMeans clustering.
    
    Args:
        current_trace_data (np.ndarray): 1D array of current values (e.g., in pA).
                                         It's recommended to use a subset of the data
                                         for clustering if the trace is very long,
                                         or use the full trace for a complete picture.
        n_states (int): The number of expected conductance states.
        initial_centroids (np.ndarray, optional): 1D array of initial centroid values.
                                                  Must have a length equal to n_states.
                                                  Defaults to None, in which case KMeans++ is used.
        random_state (int): Seed for reproducibility of KMeans. Set to None for random initialization.
    
    Returns:
        tuple: (centroids, labels)
            centroids (np.ndarray): 1D array of the mean current for each state, sorted.
            labels (np.ndarray): 1D array of state labels (0 to n_states-1) for each data point
                                 in current_trace_data, mapped to sorted centroids.
    """
    if not isinstance(current_trace_data, np.ndarray) or current_trace_data.ndim > 1:
        current_trace_data = np.asarray(current_trace_data).flatten()
    
    if initial_centroids is not None and len(initial_centroids) != n_states:
        raise ValueError(f"Length of initial_centroids ({len(initial_centroids)}) must equal n_states ({n_states}).")
    
    # Reshape for KMeans (expects 2D array: n_samples, n_features)
    X = current_trace_data.reshape(-1, 1)

    # Initialize KMeans. If initial_centroids are provided, use them. Otherwise, use KMeans++.
    init_param = initial_centroids.reshape(-1, 1) if initial_centroids is not None else 'k-means++'
    kmeans = KMeans(n_clusters=n_states, init=init_param, random_state=random_state, n_init=1)
    
    # Fit KMeans to the data
    kmeans.fit(X)

    # Get centroids and sort them for consistent state ordering (e.g., 0 = lowest current)
    centroids = np.sort(kmeans.cluster_centers_.flatten())
    
    # Predict labels for all data points
    raw_labels = kmeans.predict(X)

    # Map the raw labels to correspond to the sorted centroids
    original_centroids = kmeans.cluster_centers_.flatten()
    centroid_label_pairs = [(original_centroids[i], i) for i in range(n_states)]
    sorted_pairs = sorted(centroid_label_pairs, key=lambda x: x[0])
    original_to_new_label_map = {pair[1]: new_label for new_label, pair in enumerate(sorted_pairs)}
    
    # Apply the remapping to the predicted labels
    labels = np.array([original_to_new_label_map[label] for label in raw_labels])

    return centroids, labels

# --- Pilot Test Script (Updated with Dynamic Correction and Log-Scale Plot) ---
if __name__ == "__main__":
    # >>> IMPORTANT: REPLACE THIS WITH THE ACTUAL PATH TO YOUR TEST .ATF FILE <<<
    # Example: test_filepath = "C:/Users/YourUser/Documents/nanopore_data/F427Y_Tyr_Test.atf"
    test_filepath = "/Users/bakrantz/Documents/python/database/raw_data/PA_F427Y/guesthost_Tyr/11d05001-guesthost_Tyr-70_mV-F427Y-600_Hz-rpt_1.atf"
    
    try:
        print(f"Loading data from: {test_filepath}")
        times, current, voltage, header = load_atf(test_filepath)
        print(f"Successfully loaded {len(current)} data points.")

        if len(times) > 1:
            sampling_rate_hz = 1 / np.mean(np.diff(times))
            print(f"Detected sampling rate: {sampling_rate_hz:.2f} Hz")
        else:
            print("Not enough time points to determine sampling rate.")
            sampling_rate_hz = 0.0

        # --- Dynamic Baseline Correction ---
        # Call the new function to dynamically correct for baseline drift
        print("\nApplying dynamic baseline correction...")
        # You can now tune these parameters further
        corrected_current, drift_vector = dynamic_baseline_correction(current, window_size=4000, threshold_std_dev=1.0)
        
        # --- Test identify_conductance_states ---
        n_states_to_test = 4
        print(f"\nIdentifying {n_states_to_test} conductance states on corrected data...")
        
        # --- NEW: Define initial centroids based on your observations ---
        # These are just initial guesses; KMeans will refine them.
        # Based on your image, we can try to guide it to the baseline (~0 pA),
        # the two middle states, and the small, highest-current state (~5.5 pA).
        # You can adjust these values as needed.
        initial_centroids_guess = np.array([-0.07, 2.2, 3.2, 5.4])
        
        centroids, labels = identify_conductance_states(corrected_current, n_states=n_states_to_test, initial_centroids=initial_centroids_guess)
        
        print(f"Identified Centroids (pA): {centroids}")
        
        # --- Visualize the results with log-scale y-axis ---
        
        # Create a figure with two subplots stacked vertically
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Top subplot: Raw Current Data with Drift Curve
        ax1.plot(times, current, 'b-', label='Raw Current Data', alpha=0.7)
        ax1.plot(times, drift_vector, 'r-', label='Computed Drift Curve', linewidth=2)
        ax1.set_title(f'Raw Current Data and Computed Baseline Drift\n(from {os.path.basename(test_filepath)})')
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
        
        plt.title(f'Log-Scale Current Histogram with {n_states_to_test} Conductance States\n(from {os.path.basename(test_filepath)})')
        plt.xlabel('Current (pA)')
        plt.ylabel('Log(Count)')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.show()

        # Optional: Display counts per state
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\nCounts per State:")
        for label, count in zip(unique_labels, counts):
            print(f"  State {label} (Centroid: {centroids[label]:.2f} pA): {count} data points ({count/len(corrected_current)*100:.2f}%)")

    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
