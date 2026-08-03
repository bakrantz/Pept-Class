import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys
from scipy.optimize import curve_fit

# Custom classes are required to access and process data in databases
# Assuming the database directory is in the parent of the directory holding the training script
common_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if common_parent_dir not in sys.path:
    sys.path.insert(0, common_parent_dir)
try:
    from database.PeptideDatabase import PeptideData, PeptideDatabase
    print("Successfully imported database classes.")
except ImportError as e:
    print(f"Error importing database classes: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1) # Exit if essential imports fail

def load_stream(csv_filepath):
    """
    Loads a CSV file containing raw translocation event data, extracts and scales
    the current, and extracts the state labels.

    Args:
        csv_filepath (str): The path to the input CSV file. Expected columns:
                            'Time', 'Current', 'State'.

    Returns:
        tuple: raw_times (numpy array), scaled_raw_current (numpy array), raw_states (numpy array).
               Returns empty arrays if the file cannot be loaded or is empty.
    """
    print(f"Loading data from {csv_filepath}...")
    try:
        # (1) Load/read csv file into pandas dataframe
        df = pd.read_csv(csv_filepath)

        # Check for expected columns
        if 'Time' not in df.columns or 'Current' not in df.columns or 'State' not in df.columns:
            print(f"Error: CSV file '{csv_filepath}' must contain 'Time', 'Current', and 'State' columns.")
            return np.array([]), np.array([]), np.array([])

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

def segment_translocations(scaled_raw_current, raw_states, sampling_rate_hz=1000, min_duration_ms=None):
    """
    Segments the raw state sequence and scaled current trace into individual
    translocation events based on transitions out of and back into the open state.
    Handles edge cases and applies an optional minimum duration filter.

    Args:
        scaled_raw_current (numpy array): Scaled current trace.
        raw_states (numpy array): State labels for each time point.
        sampling_rate_hz (int): The sampling rate of the data in Hz.
        min_duration_ms (float, optional): Minimum event duration in milliseconds.
                                            Events shorter than this will be excluded.
                                            If None, no duration filtering is applied.
    Returns:
        tuple: event_currents (list of numpy arrays),
               state_sequences (list of lists),
               open_state (int)
               Returns empty lists/0 if no valid events are found.
    """
    print("\nSegmenting translocation events...")

    if raw_states.size == 0:
        print("Raw states data is empty, cannot segment.")
        return [], [], 0

    open_state = np.max(raw_states) # The highest observed integer value state is assumed to be the open state.
    print(f"Assuming open state corresponds to state label: {open_state}")

    event_currents = []
    state_sequences = []

    # Find indices of all open states
    open_state_indices = np.where(raw_states == open_state)[0]

    if open_state_indices.size < 2: # Need at least two open states to potentially bound an event
        print("Not enough occurrences of the open state to define complete events. Returning empty.")
        return [], [], open_state

    # start_processing_index and end_processing_index define the bounds within which events can start/end.
    # We need to consider the full trace to find potential events.
    # The crucial part is that an event must START with a non-open state and END with an open state.
    
    # Initialize current_index to start searching from the beginning of the states array
    current_index = 0

    # Calculate minimum length in time points if filter is active
    min_length_timepoints = 0
    if min_duration_ms is not None:
        min_length_timepoints = int(min_duration_ms * sampling_rate_hz / 1000.0)
        # Ensure min_length_timepoints is at least 1 for valid durations, unless min_duration_ms is 0
        if min_length_timepoints == 0 and min_duration_ms > 0:
            min_length_timepoints = 1
        print(f"Applying minimum event duration filter: {min_duration_ms} ms ({min_length_timepoints} timepoints)")


    while current_index < len(raw_states):
        # 1. Skip over initial open states until a non-open state is found
        while current_index < len(raw_states) and raw_states[current_index] == open_state:
            current_index += 1

        # If current_index is now pointing to a non-open state (potential event start)
        if current_index < len(raw_states) and raw_states[current_index] != open_state:
            event_start_index = current_index
            
            # 2. Find the end of the event (first return to open state AFTER event_start_index)
            # Start search_end_index from the current event_start_index + 1
            search_end_index = event_start_index + 1
            while search_end_index < len(raw_states) and raw_states[search_end_index] != open_state:
                search_end_index += 1

            # Check if an open state was found to close the event within the trace
            if search_end_index < len(raw_states) and raw_states[search_end_index] == open_state:
                event_end_index = search_end_index

                # Ensure the event actually has duration (event_start_index must be less than event_end_index)
                if event_start_index < event_end_index:
                    segmented_state_sequence = raw_states[event_start_index : event_end_index].tolist()
                    segmented_current_trace = scaled_raw_current[event_start_index : event_end_index]

                    # Ensure the segment contains at least one non-open state (this should be true by logic if event_start_index is a non-open state)
                    if any(state != open_state for state in segmented_state_sequence):
                        event_length_timepoints = len(segmented_state_sequence)

                        # Apply Minimum Duration Filter
                        if min_duration_ms is not None and event_length_timepoints < min_length_timepoints:
                            current_index = event_end_index # Skip this short event and continue search from its end
                            continue # Go to the next outer loop iteration
                        else:
                            # Event passes filter or no filter applied, append it
                            state_sequences.append(segmented_state_sequence)
                            event_currents.append(segmented_current_trace)
                            # print(f"Found event from index {event_start_index} to {event_end_index-1}, length {event_length_timepoints}") # Debugging line
                    
                    current_index = event_end_index # Advance to the closing open state to find the next event
                else: # This path implies event_start_index == event_end_index, a 0-length event, which shouldn't happen with correct logic
                    current_index += 1 # Just in case, advance to prevent infinite loop
            else: # No closing open state found for the current potential event (event runs to end of trace)
                break # Exit the loop, incomplete event at end of trace

        else: # current_index reached end of raw_states while in open state, or after processing all events
            break # Exit the main while loop


    print(f"Found {len(state_sequences)} translocation events (after filtering).")

    return event_currents, state_sequences, open_state

def _process_single_event(state_sequence, open_state):
    """
    Processes a single state sequence to calculate dwell times for each transition.

    Args:
        state_sequence (list): A list representing the state sequence for a single
                               translocation event.
        open_state (int): The label for the fully open channel state.

    Returns:
        dict: A dictionary where keys are (from_state, to_state) tuples and
              values are lists of dwell times in points.
    """
    dwells_by_transition = {}
    current_state = state_sequence[0]
    current_dwell = 0

    # The key step: append the open_state to ensure the last transition is captured
    # as the event returns to the baseline.
    extended_sequence = state_sequence + [open_state]

    for i in range(len(extended_sequence)):
        if extended_sequence[i] == current_state:
            current_dwell += 1
        else:
            from_state = current_state
            to_state = extended_sequence[i]
            
            transition = (from_state, to_state)
            if transition not in dwells_by_transition:
                dwells_by_transition[transition] = []
            
            # Record the dwell time in points
            dwells_by_transition[transition].append(current_dwell)
            
            # Reset for the new state
            current_state = to_state
            current_dwell = 1
            
    return dwells_by_transition

def get_all_dwells(list_of_state_sequences, open_state, time_sampling):
    """
    Aggregates dwell times from a list of state sequences into a single transition
    matrix.

    Args:
        list_of_state_sequences (list): A list of lists, where each inner list
                                        is a state sequence for a translocation event.
        open_state (int): The label for the fully open channel state.
        time_sampling (float): The time sampling of the raw data in Hertz (Hz).

    Returns:
        numpy.ndarray: A matrix of lists, where each position [i, j] contains a list
                       of all observed dwell times (in seconds) for a transition from
                       state i to state j.
    """
    n = open_state + 1
    dwell_matrix = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            dwell_matrix[i, j] = []

    if not list_of_state_sequences:
        print("Warning: Input list of state sequences is empty. Returning an empty matrix.")
        return dwell_matrix
        
    for state_sequence in list_of_state_sequences:
        if not state_sequence:
            continue
        
        dwells_dict = _process_single_event(state_sequence, open_state)
        
        # Aggregate the results into the main dwell matrix
        for (from_state, to_state), dwells_in_points in dwells_dict.items():
            dwells_in_seconds = [d / time_sampling for d in dwells_in_points]
            dwell_matrix[from_state, to_state].extend(dwells_in_seconds)
            
    return dwell_matrix

def ln_survival_func(data):
    """
    Calculates the natural logarithm of the survival function of a given dataset.

    Args:
        data (list or numpy.ndarray): The input data (dwell times in seconds).

    Returns:
        tuple: A tuple containing the sorted data and ln of the survival function.
               Returns (None, None) if the input data is empty.
    """
    if not isinstance(data, (list, np.ndarray)) or len(data) == 0:
        return None, None
        
    sorted_data = np.sort(data)
    # Use the empirical cumulative distribution function (ECDF) to calculate survival
    # The `+1` ensures we avoid log(0)
    survival = 1 - (np.arange(1, len(sorted_data) + 1) / len(sorted_data))
    
    # Filter out zeros from the survival function to avoid log(0)
    non_zero_survival_indices = survival > 0
    sorted_data_filtered = sorted_data[non_zero_survival_indices]
    survival_filtered = survival[non_zero_survival_indices]
    
    ln_survival_function = np.log(survival_filtered)
    
    return sorted_data_filtered, ln_survival_function

def one_exp_model_log(t, A, tau):
    """
    Single exponential decay model for the survival function, transformed to log scale.
    S(t) = A*exp(-t/tau)
    ln(S(t)) = ln(A) - t/tau
    
    Args:
        t (numpy.ndarray): Time values.
        A (float): Amplitude or pre-exponential factor.
        tau (float): Time constant.
    """
    # Use np.maximum to prevent log of zero or negative A
    return np.log(np.maximum(1e-10, A)) - t / tau

def get_ln_survival_and_fit_matrix_1exp(dwell_matrix):
    """
    Processes a dwell time matrix to calculate the ln survival function for each
    transition and performs a single exponential fit for parameters A and tau.

    Args:
        dwell_matrix (numpy.ndarray): A matrix of lists containing dwell times in seconds.

    Returns:
        numpy.ndarray: A new matrix of dictionaries. Each dictionary contains:
                         - 'times': The sorted dwell times.
                         - 'ln_survival': The ln of the survival function.
                         - 'fit': A dictionary with 'A', 'tau', 'r_squared', and their errors.
                         - 'residuals': The residuals of the fit.
                         - For failed fits, fit parameters are np.nan.
    """
    rows, cols = dwell_matrix.shape
    results_matrix = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            dwells = dwell_matrix[i, j]
            
            if dwells is None or len(dwells) < 2:
                results_matrix[i, j] = None
                continue

            # Assuming ln_survival_func exists and is defined elsewhere
            times, ln_survival = ln_survival_func(dwells)
            
            if times is None or len(times) < 2:
                results_matrix[i, j] = None
                continue
            
            try:
                # Set initial guess parameters and bounds for A and tau.
                p0 = [1.0, 0.01]  # Initial guess for A, tau
                bounds = ([1e-9, 1e-6], [1.1, 10.0]) # Bounds: A must be > 0, tau > 0

                popt, pcov = curve_fit(one_exp_model_log, times, ln_survival, p0=p0, bounds=bounds)
                A, tau = popt
                perr = np.sqrt(np.diag(pcov)) # Standard deviation errors
                
                # Calculate R-squared to quantify the quality of the fit
                y_predicted = one_exp_model_log(times, A, tau)
                ss_total = np.sum((ln_survival - np.mean(ln_survival)) ** 2)
                ss_residual = np.sum((ln_survival - y_predicted) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 1.0

                # Calculate Bayesian Information Criterion (BIC) 
                n = len(times) 
                k = 2 # Number of parameters in model
                bic = np.nan # Default to nan

                if np.isfinite(ss_residual):
                    # Safeguard against log(0) for a perfect fit
                    if ss_residual > 0:
                        # Use the standard formula with ss_residual/n
                        bic = n * np.log(ss_residual / n) + k * np.log(n)
                    else:
                        # Perfect fit case: assign a very large negative number
                        # by using a tiny floor value for log argument.
                        bic = n * np.log(1e-12) + k * np.log(n)

                
                # Calculate the residuals
                residuals = ln_survival - y_predicted
                
                results_matrix[i, j] = {
                    'times': times,
                    'ln_survival': ln_survival,
                    'fit': {
                        'A': A,
                        'tau': tau,
                        'r_squared': r_squared,
                        'bic': bic,
                        'A_err': perr[0],
                        'tau_err': perr[1],
                        'tau_mean': tau,
                        'tau_mean_err': perr[1] 
                    },
                    'residuals': residuals
                }
            except RuntimeError as e:
                print(f"1-exp fitting failed for transition {i} -> {j}: {e}")
                # Store original data but indicate failed fit with NaN values
                results_matrix[i, j] = {
                    'times': times,
                    'ln_survival': ln_survival,
                    'fit': {
                        'A': np.nan,
                        'tau': np.nan,
                        'r_squared': np.nan,
                        'bic': np.nan,
                        'A_err': np.nan,
                        'tau_err': np.nan,
                        'tau_mean': np.nan,
                        'tau_mean_err': np.nan
                    },
                    'residuals': np.full_like(times, np.nan)
                }
                
    return results_matrix

def two_exp_model_log(t, A1, tau1, A2, tau2):
    """
    Double exponential decay model for the survival function, transformed to log scale.
    S(t) = A1*exp(-t/tau1) + A2*exp(-t/tau2)
    ln(S(t)) = ln(A1*exp(-t/tau1) + A2*exp(-t/tau2))
    
    Args:
        t (numpy.ndarray): Time values.
        A1 (float): Amplitude of the first component.
        tau1 (float): Time constant of the first component.
        A2 (float): Amplitude of the second component.
        tau2 (float): Time constant of the second component.
    """
    # Use np.maximum to prevent log of zero or negative values
    return np.log(np.maximum(1e-10, A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)))

def get_ln_survival_and_fit_matrix_2exp(dwell_matrix):
    """
    Processes a dwell time matrix to calculate the ln survival function for each
    transition and performs a double exponential fit.
    Adds standard deviation error values for the fit parameters.

    Args:
        dwell_matrix (numpy.ndarray): A matrix of lists containing dwell times in seconds.

    Returns:
        numpy.ndarray: A new matrix of dictionaries. Each dictionary contains:
                         - 'times': The sorted dwell times.
                         - 'ln_survival': The ln of the survival function.
                         - 'fit': A dictionary with 'A1', 'tau1', 'A2', 'tau2', 'r_squared', and their errors.
                         - 'residuals': The residuals of the fit.
                         - For failed fits, fit parameters are np.nan.
    """
    rows, cols = dwell_matrix.shape
    results_matrix = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            dwells = dwell_matrix[i, j]
            
            if dwells is None or len(dwells) < 5: # Need more points for 4-parameter fit
                results_matrix[i, j] = None
                continue

            times, ln_survival = ln_survival_func(dwells)
            
            if times is None or len(times) < 5:
                results_matrix[i, j] = None
                continue
            try:
                # Set initial guess parameters.
                p0 = [0.5, 0.005, 0.5, 0.05] # A1, tau1, A2, tau2
                
                # Bounds for the parameters.
                bounds = ([0.0, 1e-6, 0.0, 1e-6], [1.0, 1.0, 1.0, 1.0])
                
                popt, pcov = curve_fit(two_exp_model_log, times, ln_survival, p0=p0, bounds=bounds)
                perr = np.sqrt(np.diag(pcov)) # Standard deviation errors
                
                y_predicted = two_exp_model_log(times, *popt)

                # Mean lifetime assumes amplitudes sum to 1
                A1, tau1, A2, tau2 = popt
                tau_mean = (A1 * tau1) + (A2 * tau2)
                # Error propagation using the covariance matrix
                # Gradient of tau_mean w.r.t. [A1, tau1, A2, tau2]
                J = np.array([tau1, A1, tau2, A2]) 
                # Variance = J * C * J^T
                tau_mean_var = J.T @ pcov @ J
                tau_mean_err = np.sqrt(tau_mean_var) if tau_mean_var > 0 else np.nan
                
                # Calculate R-squared
                ss_total = np.sum((ln_survival - np.mean(ln_survival)) ** 2)
                ss_residual = np.sum((ln_survival - y_predicted) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 1.0

                # Calculate Bayesian Information Criterion (BIC) 
                n = len(times) 
                k = 4 # Number of parameters in model
                bic = np.nan # Default to nan

                if np.isfinite(ss_residual):
                    # Safeguard against log(0) for a perfect fit
                    if ss_residual > 0:
                        # Use the standard formula with ss_residual/n
                        bic = n * np.log(ss_residual / n) + k * np.log(n)
                    else:
                        # Perfect fit case: assign a very large negative number
                        # by using a tiny floor value for log argument.
                        bic = n * np.log(1e-12) + k * np.log(n)
                        
                # Calculate the residuals
                residuals = ln_survival - y_predicted
                
                results_matrix[i, j] = {
                    'times': times,
                    'ln_survival': ln_survival,
                    'fit': {
                        'A1': popt[0],
                        'tau1': popt[1],
                        'A2': popt[2],
                        'tau2': popt[3],
                        'r_squared': r_squared,
                        'bic': bic,
                        'A1_err': perr[0],
                        'tau1_err': perr[1],
                        'A2_err': perr[2],
                        'tau2_err': perr[3],
                        'tau_mean': tau_mean,
                        'tau_mean_err': tau_mean_err
                    },
                    'residuals': residuals
                }

            except RuntimeError as e:
                print(f"2-exp fitting failed for transition {i} -> {j}: {e}")
                # **MODIFIED BLOCK**: Store original data but indicate failed fit with NaN values
                results_matrix[i, j] = {
                    'times': times,
                    'ln_survival': ln_survival,
                    'fit': {
                        'A1': np.nan,
                        'tau1': np.nan,
                        'A2': np.nan,
                        'tau2': np.nan,
                        'r_squared': np.nan,
                        'bic': np.nan,
                        'A1_err': np.nan,
                        'tau1_err': np.nan,
                        'A2_err': np.nan,
                        'tau2_err': np.nan,
                        'tau_mean': np.nan,
                        'tau_mean_err': np.nan
                    },
                    'residuals': np.full_like(times, np.nan)
                }
    
    return results_matrix

def three_exp_model_log(t, A1, tau1, A2, tau2, A3, tau3):
    """
    Triple exponential decay model for the survival function, transformed to log scale.
    S(t) = A1*exp(-t/tau1) + A2*exp(-t/tau2) + A3*exp(-t/tau3)
    ln(S(t)) = ln(A1*exp(-t/tau1) + A2*exp(-t/tau2) + A3*exp(-t/tau3))
    
    Args:
        t (numpy.ndarray): Time values.
        A1 (float): Amplitude of the first component.
        tau1 (float): Time constant of the first component.
        A2 (float): Amplitude of the second component.
        tau2 (float): Time constant of the second component.
        A3 (float): Amplitude of the third component.
        tau3 (float): Time constant of the third component.
    """
    # Use np.maximum to prevent log of zero or negative values
    return np.log(np.maximum(1e-10, A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + A3 * np.exp(-t / tau3)))

def get_ln_survival_and_fit_matrix_3exp(dwell_matrix):
    """
    Processes a dwell time matrix to calculate the ln survival function for each
    transition and performs a triple exponential fit.
    Adds standard deviation error values for the fit parameters.

    Args:
        dwell_matrix (numpy.ndarray): A matrix of lists containing dwell times in seconds.

    Returns:
        numpy.ndarray: A new matrix of dictionaries. Each dictionary contains:
                         - 'times': The sorted dwell times.
                         - 'ln_survival': The ln of the survival function.
                         - 'fit': A dictionary with 'A1', 'tau1', 'A2', 'tau2', 'A3', 'tau3', 'r_squared', and their errors.
                         - 'residuals': The residuals of the fit.
                         - For failed fits, fit parameters are np.nan.
    """
    rows, cols = dwell_matrix.shape
    results_matrix = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            dwells = dwell_matrix[i, j]
            
            # Need more points for a 6-parameter fit. Adjust this number as needed.
            if dwells is None or len(dwells) < 10:
                results_matrix[i, j] = None
                continue

            times, ln_survival = ln_survival_func(dwells)
            
            if times is None or len(times) < 10:
                results_matrix[i, j] = None
                continue
            
            try:
                # Set initial guess parameters.
                p0 = [0.33, 0.005, 0.33, 0.05, 0.34, 0.5] # Amplitudes should sum to 1.
                
                # Bounds for the parameters.
                bounds = ([0.0, 1e-6, 0.0, 1e-6, 0.0, 1e-6], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                
                popt, pcov = curve_fit(three_exp_model_log, times, ln_survival, p0=p0, bounds=bounds)
                perr = np.sqrt(np.diag(pcov)) # Standard deviation errors
                
                y_predicted = three_exp_model_log(times, *popt)

                # Calculate tau_mean and its error ---
                A1, tau1, A2, tau2, A3, tau3 = popt
                tau_mean = (A1 * tau1) + (A2 * tau2) + (A3 * tau3)
                # Gradient of tau_mean w.r.t. [A1, tau1, A2, tau2, A3, tau3]
                J = np.array([tau1, A1, tau2, A2, tau3, A3])
                tau_mean_var = J.T @ pcov @ J
                tau_mean_err = np.sqrt(tau_mean_var) if tau_mean_var > 0 else np.nan
                
                # Calculate R-squared
                ss_total = np.sum((ln_survival - np.mean(ln_survival)) ** 2)
                ss_residual = np.sum((ln_survival - y_predicted) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 1.0

                # Calculate Bayesian Information Criterion (BIC) 
                n = len(times) 
                k = 6 # Number of parameters in model
                bic = np.nan # Default to nan

                if np.isfinite(ss_residual):
                    # Safeguard against log(0) for a perfect fit
                    if ss_residual > 0:
                        # Use the standard formula with ss_residual/n
                        bic = n * np.log(ss_residual / n) + k * np.log(n)
                    else:
                        # Perfect fit case: assign a very large negative number
                        # by using a tiny floor value for log argument.
                        bic = n * np.log(1e-12) + k * np.log(n)
                
                # Calculate the residuals
                residuals = ln_survival - y_predicted
                
                results_matrix[i, j] = {
                    'times': times,
                    'ln_survival': ln_survival,
                    'fit': {
                        'A1': popt[0],
                        'tau1': popt[1],
                        'A2': popt[2],
                        'tau2': popt[3],
                        'A3': popt[4],
                        'tau3': popt[5],
                        'r_squared': r_squared,
                        'bic': bic,
                        'A1_err': perr[0],
                        'tau1_err': perr[1],
                        'A2_err': perr[2],
                        'tau2_err': perr[3],
                        'A3_err': perr[4],
                        'tau3_err': perr[5],
                        'tau_mean': tau_mean,
                        'tau_mean_err': tau_mean_err
                    },
                    'residuals': residuals
                }

            except RuntimeError as e:
                print(f"3-exp fitting failed for transition {i} -> {j}: {e}")
                # **MODIFIED BLOCK**: Store original data but indicate failed fit with NaN values
                results_matrix[i, j] = {
                    'times': times,
                    'ln_survival': ln_survival,
                    'fit': {
                        'A1': np.nan,
                        'tau1': np.nan,
                        'A2': np.nan,
                        'tau2': np.nan,
                        'A3': np.nan,
                        'tau3': np.nan,
                        'r_squared': np.nan,
                        'bic': np.nan,
                        'A1_err': np.nan,
                        'tau1_err': np.nan,
                        'A2_err': np.nan,
                        'tau2_err': np.nan,
                        'A3_err': np.nan,
                        'tau3_err': np.nan,
                        'tau_mean': np.nan,
                        'tau_mean_err': np.nan
                    },
                    'residuals': np.full_like(times, np.nan)
                }
    
    return results_matrix

def plot_ln_survival_and_residuals(results_matrix, nanopore_name, peptide_name, model_type, plot_dir='./plots/'):
    """
    Generates and saves two pages of plots for a given model type:
    1. Log-survival plots with model fits.
    2. Residuals of the fits.

    Args:
        results_matrix (numpy.ndarray): The matrix of dictionaries from the fitting function.
        nanopore_name (str): Name of the nanopore for plot labels and filenames.
        peptide_name (str): Name of the peptide for plot labels and filenames.
        model_type (str): '1-exp', '2-exp', or '3-exp' for plotting logic.
        plot_dir (str): The directory to save the plots.
    """
    # Create the directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    rows, cols = results_matrix.shape
    
    # Filter out empty transitions
    data_transitions = []
    for i in range(rows):
        for j in range(cols):
            if results_matrix[i, j] is not None:
                data_transitions.append((i, j, results_matrix[i, j]))

    if not data_transitions:
        print(f"No transitions with sufficient data to plot for {model_type} model.")
        return

    num_plots = len(data_transitions)
    
    # Dynamically determine a reasonable grid size
    grid_size = int(np.ceil(np.sqrt(num_plots)))
    fig_size = (5 * grid_size, 4 * grid_size)

    # --- Page 1: Log-Survival Plots with Fits ---
    fig1, axes1 = plt.subplots(grid_size, grid_size, figsize=fig_size)
    axes1 = np.array(axes1).flatten()
        
    for idx, (i, j, result) in enumerate(data_transitions):
        ax = axes1[idx]
        
        times = result['times']
        ln_survival = result['ln_survival']
        fit = result['fit']
        
        # Plot the raw data points
        ax.plot(times, ln_survival, 'o', label='Data', markersize=4)
        
        # Plot the best-fit curve, checking for failed fits (np.nan)
        if model_type == '1-exp':
            if not np.isnan(fit['A']): # Check if fit was successful
                y_predicted = one_exp_model_log(times, fit['A'], fit['tau'])
                ax.plot(times, y_predicted, '-', color='r', linewidth=2, label='1-Exp Fit')
        
        elif model_type == '2-exp':
            if not np.isnan(fit['A1']): # Check if fit was successful
                y_predicted = two_exp_model_log(times, fit['A1'], fit['tau1'], fit['A2'], fit['tau2'])
                ax.plot(times, y_predicted, '-', color='r', linewidth=2, label='2-Exp Fit')

        elif model_type == '3-exp':
            if not np.isnan(fit['A1']): # Check if fit was successful
                y_predicted = three_exp_model_log(times, fit['A1'], fit['tau1'], fit['A2'], fit['tau2'], fit['A3'], fit['tau3'])
                ax.plot(times, y_predicted, '-', color='r', linewidth=2, label='3-Exp Fit')
        
        ax.set_title(f'State {i} $\\rightarrow$ State {j}', fontsize=10)
        ax.set_xlabel('Dwell Time ($s$)', fontsize=8)
        ax.set_ylabel('ln(Survival)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for k in range(num_plots, len(axes1)):
        axes1[k].axis('off')
        
    fig1.suptitle(f'{model_type.upper()} Log-Survival Plots for {peptide_name} peptide via {nanopore_name} nanopore', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    survival_file = f'{nanopore_name}_{peptide_name}_{model_type}_log_survival_plots.png'
    fig1.savefig(os.path.join(plot_dir, survival_file))
    print(f"{model_type.upper()} log-survival plots saved to '{os.path.join(plot_dir, survival_file)}'")
    plt.close(fig1)

    # --- Page 2: Residuals Plots ---
    fig2, axes2 = plt.subplots(grid_size, grid_size, figsize=fig_size)
    axes2 = np.array(axes2).flatten()
        
    for idx, (i, j, result) in enumerate(data_transitions):
        ax = axes2[idx]

        times = result['times']
        residuals = result['residuals']
        
        # Only plot if residuals are not all NaN (i.e., fit was successful)
        if not np.all(np.isnan(residuals)):
            ax.plot(times, residuals, 'o', markersize=4)
            ax.axhline(0, color='r', linestyle='--')
        
        ax.set_title(f'State {i} $\\rightarrow$ State {j} Residuals', fontsize=10)
        ax.set_xlabel('Dwell Time ($s$)', fontsize=8)
        ax.set_ylabel('Residuals', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
    # Hide any unused subplots
    for k in range(num_plots, len(axes2)):
        axes2[k].axis('off')
        
    fig2.suptitle(f'Residuals of {model_type.upper()} Fits for {peptide_name} peptide via {nanopore_name} nanopore', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    residual_file = f'{nanopore_name}_{peptide_name}_{model_type}_residuals_plots.png'
    fig2.savefig(os.path.join(plot_dir, residual_file))
    print(f"Residuals plots saved to '{os.path.join(plot_dir, residual_file)}'")
    plt.close(fig2)


def save_results_to_csv(results_matrix, nanopore_name, peptide_name, model_type, output_filepath):
    """
    Saves the kinetic analysis results to a CSV file, including standard error values.

    Args:
        results_matrix (numpy.ndarray): Matrix of dictionaries from fitting function.
        nanopore_name (str): The name of the nanopore.
        peptide_name (str): The name of the peptide.
        model_type (str): Either '1-exp', '2-exp', or '3-exp' to determine which results to save.
        output_filepath (str): The path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Prepare a list of dictionaries for the DataFrame
    results_list = []
    
    rows, cols = results_matrix.shape
    for i in range(rows):
        for j in range(cols):
            result = results_matrix[i, j]
            if result is not None:
                row_data = {
                    'nanopore_name': nanopore_name,
                    'peptide_name': peptide_name,
                    'from_state': i,
                    'to_state': j,
                    'r_squared': result['fit']['r_squared'],
                    'bic': result['fit']['bic']
                }
                
                # Update with parameters and their errors based on model type
                if model_type == '1-exp':
                    row_data.update({
                        'A': result['fit']['A'],
                        'A_err': result['fit']['A_err'],
                        'tau': result['fit']['tau'],
                        'tau_err': result['fit']['tau_err'],
                        'tau_mean': result['fit']['tau_mean'],
                        'tau_mean_err': result['fit']['tau_mean_err']
                    })
                elif model_type == '2-exp':
                    row_data.update({
                        'A1': result['fit']['A1'],
                        'A1_err': result['fit']['A1_err'],
                        'tau1': result['fit']['tau1'],
                        'tau1_err': result['fit']['tau1_err'],
                        'A2': result['fit']['A2'],
                        'A2_err': result['fit']['A2_err'],
                        'tau2': result['fit']['tau2'],
                        'tau2_err': result['fit']['tau2_err'],
                        'tau_mean': result['fit']['tau_mean'],
                        'tau_mean_err': result['fit']['tau_mean_err']
                    })
                elif model_type == '3-exp':
                    row_data.update({
                        'A1': result['fit']['A1'],
                        'A1_err': result['fit']['A1_err'],
                        'tau1': result['fit']['tau1'],
                        'tau1_err': result['fit']['tau1_err'],
                        'A2': result['fit']['A2'],
                        'A2_err': result['fit']['A2_err'],
                        'tau2': result['fit']['tau2'],
                        'tau2_err': result['fit']['tau2_err'],
                        'A3': result['fit']['A3'],
                        'A3_err': result['fit']['A3_err'],
                        'tau3': result['fit']['tau3'],
                        'tau3_err': result['fit']['tau3_err'],
                        'tau_mean': result['fit']['tau_mean'],
                        'tau_mean_err': result['fit']['tau_mean_err']
                    })
                results_list.append(row_data)

    if not results_list:
        print(f"No {model_type} results to save for {nanopore_name} and {peptide_name}.")
        return

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    
    # Check if file exists to determine whether to write header
    if os.path.exists(output_filepath):
        df.to_csv(output_filepath, mode='a', header=False, index=False)
        print(f"Appended {model_type} results to '{output_filepath}'")
    else:
        df.to_csv(output_filepath, mode='w', header=True, index=False)
        print(f"Created and saved {model_type} results to '{output_filepath}'")

def consolidate_and_save_best_models(results_1exp, results_2exp, results_3exp, 
                                     nanopore_name, peptide_name, output_dir='./results/'):
    """
    Selects the best kinetic model for each transition based on the lowest BIC score,
    consolidates the results into a standardized format, and saves to a CSV file.

    Args:
        results_1exp (np.ndarray): Results matrix from the 1-exp fit.
        results_2exp (np.ndarray): Results matrix from the 2-exp fit.
        results_3exp (np.ndarray): Results matrix from the 3-exp fit.
        nanopore_name (str): Name of the nanopore.
        peptide_name (str): Name of the peptide.
        output_dir (str): Directory to save the consolidated CSV file.
    """
    print(f"\nConsolidating best models for {peptide_name}...")
    master_results = []
    rows, cols = results_1exp.shape

    for i in range(rows):
        for j in range(cols):
            models = {
                '1-exp': results_1exp[i, j],
                '2-exp': results_2exp[i, j],
                '3-exp': results_3exp[i, j]
            }

            # Find the best model by minimum BIC, ignoring failed fits
            best_model_name = None
            lowest_bic = float('inf')
            
            for name, result in models.items():
                if result and 'bic' in result['fit'] and np.isfinite(result['fit']['bic']):
                    if result['fit']['bic'] < lowest_bic:
                        lowest_bic = result['fit']['bic']
                        best_model_name = name

            if best_model_name is None:
                continue # Skip transitions where no model fit successfully

            best_result = models[best_model_name]
            fit = best_result['fit']

            # --- Normalize components into fast, middle, slow ---
            components = []
            if best_model_name == '1-exp':
                components.append({'tau': fit['tau'], 'A': fit['A'], 'tau_err': fit['tau_err'], 'A_err': fit['A_err']})
            
            elif best_model_name == '2-exp':
                components.append({'tau': fit['tau1'], 'A': fit['A1'], 'tau_err': fit['tau1_err'], 'A_err': fit['A1_err']})
                components.append({'tau': fit['tau2'], 'A': fit['A2'], 'tau_err': fit['tau2_err'], 'A_err': fit['A2_err']})

            elif best_model_name == '3-exp':
                components.append({'tau': fit['tau1'], 'A': fit['A1'], 'tau_err': fit['tau1_err'], 'A_err': fit['A1_err']})
                components.append({'tau': fit['tau2'], 'A': fit['A2'], 'tau_err': fit['tau2_err'], 'A_err': fit['A2_err']})
                components.append({'tau': fit['tau3'], 'A': fit['A3'], 'tau_err': fit['tau3_err'], 'A_err': fit['A3_err']})
            
            # Sort components by tau (lifetime)
            components.sort(key=lambda x: x['tau'])

            # --- Build the consolidated row for the CSV ---
            consolidated_row = {
                'peptide_name': peptide_name,
                'nanopore_name': nanopore_name,
                'transition_from': i,
                'transition_to': j,
                'best_model': best_model_name,
                'bic': fit['bic'],
                'r_squared': fit['r_squared'],
                'tau_mean': fit['tau_mean'],
                'tau_mean_err': fit['tau_mean_err'],
                'tau_fast': components[0]['tau'] if len(components) > 0 else np.nan,
                'A_fast': components[0]['A'] if len(components) > 0 else np.nan,
                'tau_fast_err': components[0]['tau_err'] if len(components) > 0 else np.nan,
                'A_fast_err': components[0]['A_err'] if len(components) > 0 else np.nan,
                'tau_middle': components[1]['tau'] if len(components) > 1 else np.nan,
                'A_middle': components[1]['A'] if len(components) > 1 else np.nan,
                'tau_middle_err': components[1]['tau_err'] if len(components) > 1 else np.nan,
                'A_middle_err': components[1]['A_err'] if len(components) > 1 else np.nan,
                'tau_slow': components[2]['tau'] if len(components) > 2 else (components[1]['tau'] if len(components) == 2 else np.nan),
                'A_slow': components[2]['A'] if len(components) > 2 else (components[1]['A'] if len(components) == 2 else np.nan),
                'tau_slow_err': components[2]['tau_err'] if len(components) > 2 else (components[1]['tau_err'] if len(components) == 2 else np.nan),
                'A_slow_err': components[2]['A_err'] if len(components) > 2 else (components[1]['A_err'] if len(components) == 2 else np.nan),
            }
            master_results.append(consolidated_row)

    if not master_results:
        print(f"No valid models to consolidate for {peptide_name}.")
        return
        
    # Create DataFrame and save to CSV
    df = pd.DataFrame(master_results)
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f'{nanopore_name}_{peptide_name}_consolidated_kinetics.csv')
    df.to_csv(output_filepath, index=False, float_format='%.6g')
    print(f"Consolidated results saved to '{output_filepath}'")

def consolidate_and_save_best_models_dbic(results_1exp, results_2exp, results_3exp,
                                          nanopore_name, peptide_name, output_dir='./results/',
                                          dbic_threshold=6):
    """
    Selects the best kinetic model for each transition using a Delta-BIC threshold
    to favor parsimony, consolidates the results, and saves to a CSV file.

    Args:
        results_1exp (np.ndarray): Results matrix from the 1-exp fit.
        results_2exp (np.ndarray): Results matrix from the 2-exp fit.
        results_3exp (np.ndarray): Results matrix from the 3-exp fit.
        nanopore_name (str): Name of the nanopore.
        peptide_name (str): Name of the peptide.
        output_dir (str): Directory to save the consolidated CSV file.
        dbic_threshold (float): The BIC difference required to justify a more complex model.
    """
    print(f"\nConsolidating best models for {peptide_name} using Delta-BIC > {dbic_threshold}...")
    master_results = []
    rows, cols = results_1exp.shape

    for i in range(rows):
        for j in range(cols):
            models_data = {
                '1-exp': results_1exp[i, j],
                '2-exp': results_2exp[i, j],
                '3-exp': results_3exp[i, j]
            }

            # --- New Model Selection Logic with Delta-BIC ---
            bics = {name: res['fit']['bic'] for name, res in models_data.items() 
                    if res and 'bic' in res['fit'] and np.isfinite(res['fit']['bic'])}

            if not bics:
                continue # Skip if no models fit successfully

            # Start with the simplest valid model as the best
            best_model_name = '1-exp' if '1-exp' in bics else (list(bics.keys())[0])

            # Check if 2-exp is a significant improvement over 1-exp
            if '1-exp' in bics and '2-exp' in bics:
                if (bics['1-exp'] - bics['2-exp']) > dbic_threshold:
                    best_model_name = '2-exp' # Upgrade to 2-exp

            # Check if 3-exp is a significant improvement over the CURRENT best model
            current_best_bic = bics[best_model_name]
            if '3-exp' in bics:
                if (current_best_bic - bics['3-exp']) > dbic_threshold:
                    best_model_name = '3-exp' # Upgrade to 3-exp
            # --- End of New Logic ---

            best_result = models_data[best_model_name]
            fit = best_result['fit']

            # --- Normalize components into fast, middle, slow ---
            components = []
            if best_model_name == '1-exp':
                components.append({'tau': fit['tau'], 'A': fit['A'], 'tau_err': fit['tau_err'], 'A_err': fit['A_err']})
            
            elif best_model_name == '2-exp':
                components.append({'tau': fit['tau1'], 'A': fit['A1'], 'tau_err': fit['tau1_err'], 'A_err': fit['A1_err']})
                components.append({'tau': fit['tau2'], 'A': fit['A2'], 'tau_err': fit['tau2_err'], 'A_err': fit['A2_err']})

            elif best_model_name == '3-exp':
                components.append({'tau': fit['tau1'], 'A': fit['A1'], 'tau_err': fit['tau1_err'], 'A_err': fit['A1_err']})
                components.append({'tau': fit['tau2'], 'A': fit['A2'], 'tau_err': fit['tau2_err'], 'A_err': fit['A2_err']})
                components.append({'tau': fit['tau3'], 'A': fit['A3'], 'tau_err': fit['tau3_err'], 'A_err': fit['A3_err']})
            
            # Sort components by tau (lifetime)
            components.sort(key=lambda x: x['tau'])

            # --- Build the consolidated row for the CSV ---
            consolidated_row = {
                'peptide_name': peptide_name,
                'nanopore_name': nanopore_name,
                'transition_from': i,
                'transition_to': j,
                'best_model': best_model_name,
                'bic': fit['bic'],
                'r_squared': fit['r_squared'],
                'tau_mean': fit['tau_mean'],
                'tau_mean_err': fit['tau_mean_err'],
                'tau_fast': components[0]['tau'] if len(components) > 0 else np.nan,
                'A_fast': components[0]['A'] if len(components) > 0 else np.nan,
                'tau_fast_err': components[0]['tau_err'] if len(components) > 0 else np.nan,
                'A_fast_err': components[0]['A_err'] if len(components) > 0 else np.nan,
                'tau_middle': components[1]['tau'] if len(components) > 1 else np.nan,
                'A_middle': components[1]['A'] if len(components) > 1 else np.nan,
                'tau_middle_err': components[1]['tau_err'] if len(components) > 1 else np.nan,
                'A_middle_err': components[1]['A_err'] if len(components) > 1 else np.nan,
                'tau_slow': components[2]['tau'] if len(components) > 2 else (components[1]['tau'] if len(components) == 2 else np.nan),
                'A_slow': components[2]['A'] if len(components) > 2 else (components[1]['A'] if len(components) == 2 else np.nan),
                'tau_slow_err': components[2]['tau_err'] if len(components) > 2 else (components[1]['tau_err'] if len(components) == 2 else np.nan),
                'A_slow_err': components[2]['A_err'] if len(components) > 2 else (components[1]['A_err'] if len(components) == 2 else np.nan),
            }
            master_results.append(consolidated_row)

    if not master_results:
        print(f"No valid models to consolidate for {peptide_name}.")
        return
        
    # Create DataFrame and save to CSV
    df = pd.DataFrame(master_results)
    os.makedirs(output_dir, exist_ok=True)
    # Add a suffix to the filename to distinguish it
    output_filepath = os.path.join(output_dir, f'{nanopore_name}_{peptide_name}_consolidated_kinetics_dbic.csv')
    df.to_csv(output_filepath, index=False, float_format='%.6g')
    print(f"Consolidated results with Delta-BIC logic saved to '{output_filepath}'")
    
if __name__ == '__main__':
    peptide_names = ["guesthost_Ala", "guesthost_Leu", "guesthost_Phe", "guesthost_Thr", "guesthost_Trp", "guesthost_TrpDL", "guesthost_Tyr"]

    nanopore_name = 'PA'
    min_duration_ms = 5

    # --- Calculate Absolute Paths for Databases ---
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Define the path to your 'database' directory
    database_dir = os.path.join(project_root, 'database')

    # Construct the full, absolute paths to your database JSON files
    raw_db_json_path = os.path.join(database_dir, 'peptide_data.json')

    db = PeptideDatabase(db_file=raw_db_json_path)
    
    for peptide_name in peptide_names:
        print(f"\n--- Processing {peptide_name} ---")
        peptide_query = {
            'experimental': True,
            'nanopore_name': nanopore_name,
            'peptide_name': peptide_name,
            'voltage': 70,
            'time_sampling': 400,
            'peptide_conc': {'$gte': 5, '$lte': 20}
        }
        result_peptide_records = db.retrieve_records(peptide_query)
        
        if not result_peptide_records:
            print(f"No records found for peptide '{peptide_name}'. Skipping.")
            continue
            
        all_state_sequences = []
        time_sampling = None
        
        for r in result_peptide_records:
            data_file = r.data_file
            data_path = r.data_path
            filepath = os.path.join(data_path, data_file)
            time_sampling = r.time_sampling
            # Corrected call to load_stream, using the data_file from the record
            raw_times, raw_current, raw_states = load_stream(filepath)

            if raw_states.size > 0:
                _, states, open_state = segment_translocations(raw_current, raw_states, sampling_rate_hz=time_sampling, min_duration_ms=min_duration_ms)
                all_state_sequences.extend(states)
            else:
                print(f"Skipping record due to empty or invalid data: {data_file}")
                
        if not all_state_sequences:
            print(f"No valid translocation events found for peptide '{peptide_name}'. Skipping kinetic analysis.")
            continue
            
        dwell_matrix = get_all_dwells(all_state_sequences, open_state, time_sampling)
        
        # --- 1-Exponential Model Analysis ---
        print("\nPerforming single exponential fit...")
        results_matrix_1exp = get_ln_survival_and_fit_matrix_1exp(dwell_matrix)

        # Save results matrix as pkl object 
        pkl_file_path = f'./results/{nanopore_name}_{peptide_name}_1-exp_results.pkl'
        with open(pkl_file_path, 'wb') as file:
            pickle.dump(results_matrix_1exp, file)
        
        # Plot and save results for the single exponential model
        plot_ln_survival_and_residuals(results_matrix_1exp, nanopore_name, peptide_name, model_type='1-exp')
        
        # Save the kinetic analysis results to a separate CSV file
        output_csv_path_1exp = f'./results/{nanopore_name}_{peptide_name}_1-exp_results.csv'
        save_results_to_csv(results_matrix_1exp, nanopore_name, peptide_name, model_type='1-exp', output_filepath=output_csv_path_1exp)
        
        # --- 2-Exponential Model Analysis ---
        print("\nPerforming double exponential fit...")
        results_matrix_2exp = get_ln_survival_and_fit_matrix_2exp(dwell_matrix)

        # Save results matrix as pkl object 
        pkl_file_path = f'./results/{nanopore_name}_{peptide_name}_2-exp_results.pkl'
        with open(pkl_file_path, 'wb') as file:
            pickle.dump(results_matrix_2exp, file)
        
        # Plot and save results for the double exponential model
        plot_ln_survival_and_residuals(results_matrix_2exp, nanopore_name, peptide_name, model_type='2-exp')
        
        # Save the kinetic analysis results to a separate CSV file
        output_csv_path_2exp = f'./results/{nanopore_name}_{peptide_name}_2-exp_results.csv'
        save_results_to_csv(results_matrix_2exp, nanopore_name, peptide_name, model_type='2-exp', output_filepath=output_csv_path_2exp)

        # --- 3-Exponential Model Analysis ---
        print("\nPerforming triple exponential fit...")
        results_matrix_3exp = get_ln_survival_and_fit_matrix_3exp(dwell_matrix)

        # Save results matrix as pkl object 
        pkl_file_path = f'./results/{nanopore_name}_{peptide_name}_3-exp_results.pkl'
        with open(pkl_file_path, 'wb') as file:
            pickle.dump(results_matrix_3exp, file)
        
        # Plot and save results for the triple exponential model
        plot_ln_survival_and_residuals(results_matrix_3exp, nanopore_name, peptide_name, model_type='3-exp')
        
        # Save the kinetic analysis results to a separate CSV file
        output_csv_path_3exp = f'./results/{nanopore_name}_{peptide_name}_3-exp_results.csv'
        save_results_to_csv(results_matrix_3exp, nanopore_name, peptide_name, model_type='3-exp', output_filepath=output_csv_path_3exp)

        # Consolidate the best exponential models per transition per peptide based on best BIC logic 
        consolidate_and_save_best_models(
            results_matrix_1exp,
            results_matrix_2exp,
            results_matrix_3exp,
            nanopore_name,
            peptide_name,
            output_dir='./results/'
        )

        # Considate by more complex model only if Delta_BIC > 6
        consolidate_and_save_best_models_dbic(
            results_matrix_1exp,
            results_matrix_2exp,
            results_matrix_3exp,
            nanopore_name,
            peptide_name,
            output_dir='./results/',
            dbic_threshold=6
        )
    
    print("\n--- All kinetic analysis and CSV saving complete. ---")
