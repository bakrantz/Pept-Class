# Functional segmentation core to preprocess conductance state-labeled current vs. time recordings
# and compute event-level and global features, storing in a list of dictionaries
# with flattened features and dynamic names for ML/DL input alongside original data.
# Also outputs categorized lists of feature names for easier subset selection.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from scipy.signal import butter, filtfilt, medfilt, bessel, sosfilt, sosfreqz
import math
import statistics
import pickle
import os
import warnings # Import warnings to handle potential issues like division by zero or nan statistics

# Suppress specific warnings that might arise from operations like division by zero or nan statistics
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice") # For np.nanmean of empty slice


# --- A. Create loading function and high-pass signal processing ---
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

# --- High-pass signal processing ---
def correct_baseline_and_drift(
    time_points: np.ndarray,
    noisy_signal: np.ndarray,
    high_pass_cutoff_frequency: float = 0.5,
    filter_order: int = 3,
    polynomial_degree: int = 2,
    apply_polynomial_correction: bool = True
) -> np.ndarray:
    """
    Applies high-pass filtering and optional polynomial fitting to correct
    baseline drift in a noisy signal.

    Args:
        time_points (np.ndarray): An array of time points corresponding to the signal.
                                  Assumed to be uniformly spaced for Nyquist calculation.
        noisy_signal (np.ndarray): The input signal containing noise and baseline drift.
        high_pass_cutoff_frequency (float): The cutoff frequency for the high-pass filter.
                                            Frequencies below this will be attenuated.
                                            Set to 0 or a negative value to disable high-pass filtering.
                                            Default is 0.5 (Hz if time_points are in seconds).
        filter_order (int): The order of the Butterworth high-pass filter. Higher orders
                            provide a steeper roll-off but can introduce more ringing.
                            Default is 3.
        polynomial_degree (int): The degree of the polynomial to fit for baseline correction.
                                Set to 0 to fit a constant, 1 for linear, 2 for quadratic, etc.
                                Default is 2.
        apply_polynomial_correction (bool): If True, a polynomial will be fitted to the
                                            high-pass filtered signal (or original signal if filtering is off)
                                            and subtracted.
                                            If False, only high-pass filtering is applied (or original signal returned).
                                            Default is True.

    Returns:
        np.ndarray: The baseline-corrected signal.

    Raises:
        ValueError: If time_points and noisy_signal do not have the same length.
        ValueError: If time_points has fewer than 2 elements (cannot calculate Nyquist).
    """
    if time_points.shape != noisy_signal.shape:
        raise ValueError("time_points and noisy_signal must have the same shape.")
    if time_points.size < 2:
        raise ValueError("time_points must contain at least 2 elements to calculate Nyquist frequency.")

    # --- New Logic to Disable Filtering and Correction ---
    # If high_pass_cutoff_frequency is <= 0 and polynomial correction is also disabled,
    # just return the original noisy signal.
    # We allow polynomial correction to apply even if high_pass_cutoff_frequency is 0,
    # in which case it will apply to the original noisy_signal.
    if high_pass_cutoff_frequency <= 0 and not apply_polynomial_correction:
        # print("High-pass filtering and polynomial correction are disabled. Returning original signal.")
        return noisy_signal
    
    # Calculate the Nyquist frequency. Assumes uniform sampling.
    sampling_rate = 1.0 / np.mean(np.diff(time_points))
    nyquist_frequency = 0.5 * sampling_rate

    # Initialize filtered_signal. It will be the noisy_signal if no high-pass filter is applied.
    filtered_signal = noisy_signal

    # 1. High-pass filtering (conditional)
    # Only apply the filter if high_pass_cutoff_frequency is positive and valid
    if high_pass_cutoff_frequency > 0 and high_pass_cutoff_frequency < nyquist_frequency:
        normalized_cutoff = high_pass_cutoff_frequency / nyquist_frequency
        
        # Design the Butterworth filter
        b, a = butter(filter_order, normalized_cutoff, btype='high')

        # Apply the filter forward and backward to avoid phase distortion
        filtered_signal = filtfilt(b, a, noisy_signal)
    elif high_pass_cutoff_frequency <= 0:
        # print(f"High-pass filtering skipped because cutoff frequency is {high_pass_cutoff_frequency}.")
        pass # filtered_signal remains noisy_signal
    else: # high_pass_cutoff_frequency >= nyquist_frequency
        print(f"Warning: High-pass cutoff frequency ({high_pass_cutoff_frequency}) is too high "
              f"(>= Nyquist frequency {nyquist_frequency}). Skipping high-pass filtering.")
        pass # filtered_signal remains noisy_signal

    # 2. Polynomial fitting and baseline correction (optional)
    if apply_polynomial_correction:
        if polynomial_degree < 0:
            raise ValueError("polynomial_degree must be a non-negative integer.")
        
        # Ensure polynomial degree is not too high for the data length
        if polynomial_degree >= len(time_points):
            temp_poly_degree = len(time_points) - 1
            if temp_poly_degree < 0: # Handle edge case of very small time_points
                temp_poly_degree = 0
            print(f"Warning: polynomial_degree ({polynomial_degree}) is too high for the "
                  f"number of data points ({len(time_points)}). Reducing degree to {temp_poly_degree}.")
            polynomial_degree = temp_poly_degree
        
        # Fit a polynomial to the signal (either raw or high-pass filtered)
        z = np.polyfit(time_points, filtered_signal, polynomial_degree)
        p = np.poly1d(z)

        # Correct the baseline by subtracting the fitted polynomial
        corrected_signal = filtered_signal - p(time_points)
    else:
        # If no polynomial correction is desired, the (potentially filtered) signal is the final output
        corrected_signal = filtered_signal

    return corrected_signal

# --- Low-pass median filtering ---
def apply_median_filter(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Applies a median filter to the input data.

    Args:
        data (np.ndarray): The input 1D numpy array of current data.
        window_size (int): The size of the median filter window. Must be odd.

    Returns:
        np.ndarray: The median-filtered data.
    """
    if window_size % 2 == 0:
        raise ValueError("Median filter window_size must be odd.")
    return medfilt(data, kernel_size=window_size)

def apply_bessel_filter(data, cutoff_hz, fs=1000.0, order=4):
    """
    Apply a 4-pole low-pass Bessel filter to the data.

    Parameters:
    data: array_like, the signal to filter.
    cutoff_hz: float, the cutoff frequency of the filter in Hz.
    fs: float, the sampling rate of the data in Hz (default: 1000 Hz).
    order: int, the order of the filter (default: 4 for 4-pole).

    Returns:
    filtered_data: array_like, the filtered signal.
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff_hz / nyq

    # Use second-order sections for numerical stability
    sos = bessel(N=order, Wn=norm_cutoff, btype='low', analog=False, output='sos', norm='phase')
    filtered_data = sosfilt(sos, data)

    return filtered_data 


# --- B. Segment raw_states and corresponding scaled_raw_current into translocation event 'state_sequences' and 'event_currents' ---
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

    open_state = np.max(raw_states) if raw_states.size > 0 else 0
    print(f"Assuming open state corresponds to state label: {open_state}")

    event_currents = []
    state_sequences = []
    current_index = 0

    open_state_indices = np.where(raw_states == open_state)[0]

    if open_state_indices.size < 2:
        print("Not enough occurrences of the open state to define complete events. Returning empty.")
        return [], [], open_state

    start_processing_index = open_state_indices[0]
    end_processing_index = open_state_indices[-1]

    if start_processing_index >= end_processing_index:
         print("First open state is at or after the last open state index. No complete events found within bounds. Returning empty.")
         return [], [], open_state

    current_index = start_processing_index

    # Calculate minimum length in time points if filter is active
    min_length_timepoints = 0
    if min_duration_ms is not None:
        min_length_timepoints = int(min_duration_ms * sampling_rate_hz / 1000.0)
        print(f"Applying minimum event duration filter: {min_duration_ms} ms ({min_length_timepoints} timepoints)")


    while current_index < end_processing_index:
        while current_index < end_processing_index and raw_states[current_index] == open_state:
            current_index += 1

        if current_index < end_processing_index and raw_states[current_index] != open_state:
            event_start_index = current_index

            search_end_index = current_index
            while search_end_index < end_processing_index and raw_states[search_end_index] != open_state:
                 search_end_index += 1

            if search_end_index <= end_processing_index and raw_states[search_end_index] == open_state:
                 event_end_index = search_end_index

                 if event_start_index < event_end_index:
                      segmented_state_sequence = raw_states[event_start_index : event_end_index].tolist()
                      segmented_current_trace = scaled_raw_current[event_start_index : event_end_index]

                      if any(state != open_state for state in segmented_state_sequence):

                           # --- Apply Minimum Duration Filter ---
                           event_length_timepoints = len(segmented_state_sequence)

                           if min_duration_ms is not None and event_length_timepoints < min_length_timepoints:
                                # Skip this event due to short duration
                                # print(f"Skipping event due to short duration: {event_length_timepoints} timepoints < {min_length_timepoints} min timepoints") # Optional debug
                                current_index = event_end_index + 1 # Still advance index past the short event
                                continue # Skip appending and go to the next outer loop iteration


                           # If filter is not applied or event passes filter, append it
                           state_sequences.append(segmented_state_sequence)
                           event_currents.append(segmented_current_trace)
                           # print(f"Found event from index {event_start_index} to {event_end_index-1}") # Debugging line


                 current_index = event_end_index + 1

            else:
                 break

        else:
            break

    print(f"Found {len(state_sequences)} translocation events (after filtering).") # Update print message

    return event_currents, state_sequences, open_state

# --- Helper functions for computing event-level features ---

# (a) Entropy
def calculate_entropy(state_sequence):
    """
    Calculates the Shannon entropy of a state sequence.

    Args:
        state_sequence (list): A list representing the state sequence (e.g., [0, 1, 0, 0, 1]).

    Returns:
        float: The Shannon entropy of the state sequence.
    """
    if not state_sequence:
        return 0.0

    counts = Counter(state_sequence)
    total_elements = len(state_sequence)
    probabilities = [count / total_elements for count in counts.values()]
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

# (b) First transition time
def calculate_first_transition_time(state_sequence_list):
    """
    Calculates the time (index) of the first transition in a state sequence.

    Args:
        state_sequence_list (list): A list representing the state sequence of a translocation event.

    Returns:
        int: The index of the first transition (0-indexed), or -1 if no transition occurs
             or if the sequence is empty.
    """
    if not state_sequence_list or len(state_sequence_list) < 2:
        return -1
    first_state = state_sequence_list[0]
    for i in range(1, len(state_sequence_list)):
        if state_sequence_list[i] != first_state:
            return i
    return -1

# (c) Number of transitions
def calculate_num_transitions(states_list):
    """
    Calculates the number of state transitions in a state sequence.

    Args:
        states_list (list): A list representing the state sequence.

    Returns:
        int: The number of transitions.
    """
    num_transitions = 0
    if len(states_list) > 1:
        for i in range(1, len(states_list)):
            if states_list[i] != states_list[i-1]:
                num_transitions += 1
    return num_transitions

# (d) Calculate Probabilities per state
def calculate_probabilities_per_state(state_sequence, open_state):
    """
    Calculates the probability of being in each state for a given state sequence.

    Args:
        state_sequence (list): A list representing the state sequence.
        open_state (int): The label for the open pore state.

    Returns:
        numpy array: A 1D numpy array of size (open_state + 1) where each element
                     is the probability of being in that state (index).
                     Returns 0.0 for states not present in the sequence.
                     Includes the probability for the open_state index (will be 0.0).
    """
    probabilities_vector = np.zeros(open_state + 1, dtype=np.float32)
    if not state_sequence:
        return probabilities_vector
    counts = Counter(state_sequence)
    total_elements = len(state_sequence)
    for state, count in counts.items():
        if 0 <= state <= open_state:
            probabilities_vector[state] = count / total_elements
    return probabilities_vector

# (e) Calculate Conductances per state
def calculate_conductances_per_state(state_sequence, event_current, open_state):
    """
    Calculates the average scaled current for each state present in the state sequence.

    Args:
        state_sequence (list): A list representing the state sequence.
        event_current (numpy array): The corresponding scaled current trace.
        open_state (int): The label for the open pore state.

    Returns:
        numpy array: A 1D numpy array of size (open_state + 1) where each element
                     is the average scaled current for that state (index).
                     Returns np.nan for states not present in the sequence (incl. open_state).
                     Returns the average current for state 0 even if near zero.
    """
    conductances_vector = np.full(open_state + 1, np.nan, dtype=np.float32)
    if not state_sequence or event_current.size == 0 or len(state_sequence) != event_current.size:
        return conductances_vector

    currents_by_state = {}
    for state in set(state_sequence):
        if 0 <= state <= open_state:
             currents_by_state[state] = []

    for state, current in zip(state_sequence, event_current):
        if state in currents_by_state:
             currents_by_state[state].append(current)

    for state, currents in currents_by_state.items():
        if currents:
            conductances_vector[state] = np.mean(currents)
    return conductances_vector

# (f) Calculate Longest Dwells per state
def calculate_longest_dwells_per_state(state_sequence, open_state):
    """
    Calculates the longest consecutive dwell time (run length) for each state
    in a state sequence.

    Args:
        state_sequence (list): A list representing the state sequence.
        open_state (int): The label for the open pore state.

    Returns:
        numpy array: A 1D numpy array of size (open_state + 1) where each element
                     is the longest dwell time for that state (index).
                     Returns 0 for states not present in the sequence (incl. open_state).
    """
    longest_dwells_vector = np.zeros(open_state + 1, dtype=np.int32)
    if not state_sequence:
        return longest_dwells_vector

    if len(state_sequence) > 0:
        current_state = state_sequence[0]
        current_dwell = 0
        for state in state_sequence:
            if state == current_state:
                current_dwell += 1
            else:
                if 0 <= current_state <= open_state:
                     longest_dwells_vector[current_state] = max(longest_dwells_vector[current_state], current_dwell)
                current_state = state
                current_dwell = 1
        if 0 <= current_state <= open_state:
             longest_dwells_vector[current_state] = max(longest_dwells_vector[current_state], current_dwell)
    return longest_dwells_vector

# (g) Calculate Average and Variance of Dwells per state transition
def calculate_avg_and_var_of_dwells_per_transition(state_sequence, open_state):
    """
    Calculates the average and variance of dwell times in state i before
    transitioning to state j. Returns 2D arrays.

    Args:
        state_sequence (list): A list representing the state sequence.
        open_state (int): The label for the open pore state.

    Returns:
        tuple: avg_dwell_matrix (numpy array), var_dwell_matrix (numpy array)
               Both are (open_state + 1) x (open_state + 1) matrices filled with np.nan.
               Entries for transitions involving open_state will be np.nan.
               Transitions between non-existent states will be np.nan.
    """
    avg_dwell_matrix = np.full((open_state + 1, open_state + 1), np.nan, dtype=np.float32)
    var_dwell_matrix = np.full((open_state + 1, open_state + 1), np.nan, dtype=np.float32)
    if not state_sequence or len(state_sequence) < 2:
        return avg_dwell_matrix, var_dwell_matrix

    dwells_by_transition = {}
    current_state = state_sequence[0]
    current_dwell = 0
    for i in range(len(state_sequence)):
        if state_sequence[i] == current_state:
            current_dwell += 1
        else:
            from_state = current_state
            to_state = state_sequence[i]
            if 0 <= from_state <= open_state and 0 <= to_state <= open_state:
                 transition = (from_state, to_state)
                 if transition not in dwells_by_transition:
                      dwells_by_transition[transition] = []
                 dwells_by_transition[transition].append(current_dwell)
            current_state = to_state
            current_dwell = 1

    for (from_state, to_state), dwells in dwells_by_transition.items():
        if dwells:
            avg_dwell = np.mean(dwells)
            var_dwell = np.var(dwells, ddof=1) if len(dwells) > 1 else 0.0
            avg_dwell_matrix[from_state, to_state] = avg_dwell
            var_dwell_matrix[from_state, to_state] = var_dwell
    return avg_dwell_matrix, var_dwell_matrix


# (h) Calculate Ratio of Probabilities per state pair
def calculate_ratio_of_probabilities_per_state_pair(probabilities_vector, open_state):
    """
    Calculates the ratio of probabilities for each pair of states (i/j).

    Args:
        probabilities_vector (numpy array): A 1D numpy array of probabilities per state.
        open_state (int): The label for the open pore state.

    Returns:
        numpy array: A (open_state + 1) x (open_state + 1) matrix where element [i, j]
                     is probabilities_vector[i] / probabilities_vector[j].
                     Handles division by zero by returning np.nan.
                     Entries for ratios involving open_state index will be based on its 0.0 probability.
    """
    ratio_matrix = np.full((open_state + 1, open_state + 1), np.nan, dtype=np.float32)
    if probabilities_vector is None or probabilities_vector.size != open_state + 1:
        print("Warning: Invalid probabilities vector provided for ratio calculation.")
        return ratio_matrix

    for i in range(open_state + 1):
        for j in range(open_state + 1):
            prob_i = probabilities_vector[i]
            prob_j = probabilities_vector[j]
            if prob_j != 0:
                ratio_matrix[i, j] = prob_i / prob_j
            else:
                pass
    return ratio_matrix

# (i) Calculate Number of States in a sequence
def calculate_num_states_in_sequence(state_sequence):
    """
    Calculates the number of unique states with non-zero probability in a state sequence.
    Since segmented sequences exclude the open state (0 after remapping), this counts
    unique non-open states present.

    Args:
        state_sequence (list): A list representing the state sequence.

    Returns:
        int: The number of unique states in the sequence. Returns 0 for empty sequence.
    """
    if not state_sequence:
        return 0
    unique_states = set(state_sequence)
    return len(unique_states)

# (j) Calculate Conductance State Observation vector
def calculate_cond_state_obs(state_sequence, open_state):
    """
    Creates a binary vector indicating which conductance states were observed
    in the state sequence.

    Args:
        state_sequence (list): A list representing the state sequence.
        open_state (int): The label for the open pore state.

    Returns:
        numpy array: A 1D numpy array of size (open_state + 1) where element i is 1
                     if state i was observed, and 0 otherwise.
                     The entry for open_state index will be 0 as it's excluded from events.
    """
    cond_state_obs_vector = np.zeros(open_state + 1, dtype=np.int32)
    if not state_sequence:
        return cond_state_obs_vector

    unique_states = set(state_sequence)
    for state in unique_states:
        if 0 <= state < open_state:
            cond_state_obs_vector[state] = 1
    return cond_state_obs_vector


# --- C. Compute event-level features from event_currents and state_sequences ---
def compute_event_level_features(event_currents, state_sequences, open_state):
    """
    Computes various event-level features for each segmented translocation event.

    Args:
        event_currents (list of numpy arrays): Scaled current traces for each event.
        state_sequences (list of lists): State sequences for each event.
        open_state (int): The label for the open pore state.

    Returns:
        list: A list of dictionaries, where each dictionary represents an event
              and contains its state sequence, current trace, and computed features.
    """
    print("\nComputing event-level features...")
    translocation_events_data = []
    if not state_sequences or not event_currents or len(state_sequences) != len(event_currents):
        print("No event sequences or currents found or mismatch in lengths. Cannot compute features.")
        return translocation_events_data

    for i in range(len(state_sequences)):
        state_sequence = state_sequences[i]
        event_current = event_currents[i]
        if not state_sequence or event_current.size == 0:
             print(f"Warning: Skipping event {i} due to empty sequence or current trace.")
             continue

        entropy = calculate_entropy(state_sequence)
        event_duration = len(state_sequence)
        num_transitions = calculate_num_transitions(state_sequence)
        first_transition_time = calculate_first_transition_time(state_sequence)
        probabilities_vector = calculate_probabilities_per_state(state_sequence, open_state)
        conductances_vector = calculate_conductances_per_state(state_sequence, event_current, open_state)
        longest_dwells_vector = calculate_longest_dwells_per_state(state_sequence, open_state)
        avg_dwell_matrix, var_dwell_matrix = calculate_avg_and_var_of_dwells_per_transition(state_sequence, open_state)
        ratio_matrix = calculate_ratio_of_probabilities_per_state_pair(probabilities_vector, open_state)
        num_states = calculate_num_states_in_sequence(state_sequence)
        cond_state_obs_vector = calculate_cond_state_obs(state_sequence, open_state)


        event_data = {
            'states': state_sequence, # original list
            'currents': event_current, # original numpy array
            'entropy': entropy, # Scalar
            'event_duration': event_duration, # Scalar
            'num_transitions': num_transitions, # Scalar
            'first_transition_time': first_transition_time, # Scalar
            'num_states': num_states, # Scalar
            # Store state-dependent features as arrays/matrices
            'cond_state_obs': cond_state_obs_vector, # Vector
            'conductance': conductances_vector, # Vector
            'probability': probabilities_vector, # Vector
            'longest_dwell': longest_dwells_vector, # Vector
            'avg_dwell': avg_dwell_matrix, # Matrix
            'var_dwell': var_dwell_matrix, # Matrix
            'ratio': ratio_matrix, # Matrix
        }
        translocation_events_data.append(event_data)

    print(f"Computed features for {len(translocation_events_data)} events.")
    return translocation_events_data


# --- D. Compute global features ---
def compute_global_features(translocation_events_data):
    """
    Computes global features aggregated across all translocation events in the stream
    and adds them to each event's dictionary.

    Args:
        translocation_events_data (list): A list of dictionaries from
                                          compute_event_level_features.

    Returns:
        list: The updated list of dictionaries with global features added to each event.
              Returns the original list if no data is provided.
    """
    print("\nComputing global features...")
    if not translocation_events_data:
        print("No event data found. Cannot compute global features.")
        return translocation_events_data

    def calculate_mean_for_key(list_of_dictionaries, key):
        values = []
        for dictionary in list_of_dictionaries:
            if key in dictionary:
                 value = dictionary[key]
                 if isinstance(value, (int, float)) and not np.isnan(value):
                    values.append(value)
        if not values:
            return np.nan
        else:
            return statistics.mean(values)

    average_event_length = calculate_mean_for_key(translocation_events_data, 'event_duration')
    average_event_entropy = calculate_mean_for_key(translocation_events_data, 'entropy')
    average_first_transition_time = calculate_mean_for_key(translocation_events_data, 'first_transition_time')
    average_num_transitions = calculate_mean_for_key(translocation_events_data, 'num_transitions')
    average_num_states = calculate_mean_for_key(translocation_events_data, 'num_states')

    print(f"Computed global averages: Length={average_event_length:.2f}, Entropy={average_event_entropy:.2f}, First Transition Time={average_first_transition_time:.2f}, Num Transitions={average_num_transitions:.2f}, Avg Num States={average_num_states:.2f}")

    all_state_sequences = [event['states'] for event in translocation_events_data]
    all_event_currents = [event['currents'] for event in translocation_events_data]

    concatenated_states = [state for seq in all_state_sequences for state in seq]
    concatenated_currents = np.concatenate(all_event_currents) if all_event_currents else np.array([])

    # Determine the global open state from the states present in the segmented events
    # This assumes the max state in segmented events is the highest intermediate state,
    # and the true open state was 1 higher. A more robust way might be to pass
    # the original open_state determined in segmentation or calculate from raw_states.
    # However, sticking to states seen in events aligns global features to that range.
    # Let's stick to calculating from concatenated_states for consistency within this function's scope.
    global_open_state_val_in_events = np.max(concatenated_states) if concatenated_states else 0
    # If you need the *true* open state label (e.g., 3 for 4-state data), you might need to pass
    # the 'open_state' value determined in segment_translocations() to this function.
    # Assuming the state labels are 0, 1, ..., open_state-1 for intermediates,
    # max(concatenated_states) gives the max intermediate state label.
    # The dimension for global features should still be based on the *true* open_state.
    # Let's pass the original 'open_state' value to this function.

    # Reworking this based on needing the true open_state for dimensioning global features
    # The function signature needs to change to accept 'open_state'


# --- D. Compute global features (Revised Signature) ---
def compute_global_features(translocation_events_data, open_state): # Accept open_state
    """
    Computes global features aggregated across all translocation events in the stream
    and adds them to each event's dictionary.

    Args:
        translocation_events_data (list): A list of dictionaries from
                                          compute_event_level_features.
        open_state (int): The label for the open pore state (used for feature dimensions).

    Returns:
        list: The updated list of dictionaries with global features added to each event.
              Returns the original list if no data is provided.
    """
    print("\nComputing global features...")
    if not translocation_events_data:
        print("No event data found. Cannot compute global features.")
        return translocation_events_data

    def calculate_mean_for_key(list_of_dictionaries, key):
        values = []
        for dictionary in list_of_dictionaries:
            if key in dictionary:
                 value = dictionary[key]
                 if isinstance(value, (int, float)) and not np.isnan(value):
                    values.append(value)
        if not values:
            return np.nan
        else:
            return statistics.mean(values)

    average_event_length = calculate_mean_for_key(translocation_events_data, 'event_duration')
    average_event_entropy = calculate_mean_for_key(translocation_events_data, 'entropy')
    average_first_transition_time = calculate_mean_for_key(translocation_events_data, 'first_transition_time')
    average_num_transitions = calculate_mean_for_key(translocation_events_data, 'num_transitions')
    average_num_states = calculate_mean_for_key(translocation_events_data, 'num_states')

    print(f"Computed global averages: Length={average_event_length:.2f}, Entropy={average_event_entropy:.2f}, First Transition Time={average_first_transition_time:.2f}, Num Transitions={average_num_transitions:.2f}, Avg Num States={average_num_states:.2f}")

    all_state_sequences = [event['states'] for event in translocation_events_data]
    all_event_currents = [event['currents'] for event in translocation_events_data]

    concatenated_states = [state for seq in all_state_sequences for state in seq]
    concatenated_currents = np.concatenate(all_event_currents) if all_event_currents else np.array([])

    # Use the provided open_state from segmentation for feature dimensioning
    global_feature_dim = open_state + 1

    # overall_probability
    # Calculate probability distribution over the entire concatenated state stream
    overall_probability = calculate_probabilities_per_state(concatenated_states, open_state) # Use open_state for dimension
    print(f"Computed global probability distribution.")

    # overall_conductance
    if len(concatenated_states) == len(concatenated_currents) and concatenated_states:
        # Use open_state for dimension
        overall_conductance = calculate_conductances_per_state(concatenated_states, concatenated_currents, open_state)
        print(f"Computed global conductance averages per state.")
    else:
        overall_conductance = np.full(global_feature_dim, np.nan, dtype=np.float32)
        print("Warning: Cannot compute global conductance due to mismatch in concatenated states/currents length or empty data.")

    # overall_ratio
    overall_ratio = calculate_ratio_of_probabilities_per_state_pair(overall_probability, open_state) # Use open_state for dimension
    print(f"Computed global probability ratios per state pair.")

    # global_num_states: Number of unique states observed globally in the *segmented events*
    global_num_states_val = len(set(concatenated_states)) if concatenated_states else 0
    print(f"Computed global number of unique states (in events): {global_num_states_val}")


    for event in translocation_events_data:
        event['average_event_length'] = average_event_length
        event['average_event_entropy'] = average_event_entropy
        event['average_first_transition_time'] = average_first_transition_time
        event['average_num_transitions'] = average_num_transitions
        event['average_num_states'] = average_num_states
        event['overall_probability'] = overall_probability
        event['overall_conductance'] = overall_conductance
        event['overall_ratio'] = overall_ratio
        event['global_num_states'] = global_num_states_val


    print(f"Added global features to {len(translocation_events_data)} events.")
    return translocation_events_data


# --- E. Prepare data for ML/DL input (list of dictionaries with flattened features) ---
def prepare_ml_dl_data(translocation_events_data, open_state): # Accept open_state
    """
    Prepares data for ML/DL input by flattening vector and matrix features,
    creating dynamic keys, and structuring as a list of dictionaries including
    original sequences and currents.

    Also generates categorized lists of feature names for easier subset selection
    in downstream models.

    Args:
        translocation_events_data (list): List of dictionaries, each representing an event
                                          (output of compute_global_features).
        open_state (int): The label for the open pore state (used for feature vector/matrix dimensions).

    Returns:
        tuple: ml_dl_events_data (list of dictionaries),
               feature_names_dict (dictionary of categorized lists of strings)
               Returns empty list and empty dict if no data.
    """
    print("\nPreparing data for ML/DL input (list of dictionaries with flattened features)...")

    if not translocation_events_data:
        print("No event data to prepare.")
        return [], {}

    # Determine feature dimensions based on the determined open_state
    max_state_label = open_state
    vector_dim = max_state_label + 1
    matrix_shape = (max_state_label + 1, max_state_label + 1)
    matrix_size = matrix_shape[0] * matrix_shape[1]

    # Define the original keys for features BEFORE flattening, categorized by scope and type
    scalar_event_keys = ['entropy', 'event_duration', 'num_transitions', 'first_transition_time', 'num_states']
    vector_event_keys = ['cond_state_obs', 'conductance', 'probability', 'longest_dwell'] # These will be flattened
    matrix_event_keys = ['avg_dwell', 'var_dwell', 'ratio'] # These will be flattened

    scalar_global_keys = [
        'average_event_length', 'average_event_entropy', 'average_first_transition_time',
        'average_num_transitions', 'average_num_states', 'global_num_states'
    ]
    vector_global_keys = ['overall_probability', 'overall_conductance'] # These will be flattened
    matrix_global_keys = ['overall_ratio'] # These will be flattened


    ml_dl_events_data = [] # List to hold the prepared dictionaries
    all_flattened_feature_names_combined = [] # To build the single list of all flattened names in order

    # Build the categorized lists of *flattened* feature names
    event_level_feature_names = []
    event_level_feature_names.extend(scalar_event_keys) # Add scalar event names

    global_feature_names = []
    global_feature_names.extend(scalar_global_keys) # Add scalar global names


    # Add names for flattened vector event features
    flattened_vector_event_names = []
    for key in vector_event_keys:
        flattened_vector_event_names.extend([f'{key}_state_{i}' for i in range(vector_dim)])
    event_level_feature_names.extend(flattened_vector_event_names) # Add to event level list
    all_flattened_feature_names_combined.extend(flattened_vector_event_names) # Add to combined list

    # Add names for flattened matrix event features
    flattened_matrix_event_names = []
    for key in matrix_event_keys:
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                 flattened_matrix_event_names.append(f'{key}_state_{i}_state_{j}')
    event_level_feature_names.extend(flattened_matrix_event_names) # Add to event level list
    all_flattened_feature_names_combined.extend(flattened_matrix_event_names) # Add to combined list


    # Add names for flattened vector global features
    flattened_vector_global_names = []
    for key in vector_global_keys:
        flattened_vector_global_names.extend([f'{key}_state_{i}' for i in range(vector_dim)])
    global_feature_names.extend(flattened_vector_global_names) # Add to global list
    all_flattened_feature_names_combined.extend(flattened_vector_global_names) # Add to combined list


    # Add names for flattened matrix global features
    flattened_matrix_global_names = []
    for key in matrix_global_keys:
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                 flattened_matrix_global_names.append(f'{key}_state_{i}_state_{j}')
    global_feature_names.extend(flattened_matrix_global_names) # Add to global list
    all_flattened_feature_names_combined.extend(flattened_matrix_global_names) # Add to combined list


    # Store categorized names in a dictionary to return
    feature_names_dict = {
        'event_level_scalar': scalar_event_keys,
        'event_level_vector_flat': flattened_vector_event_names,
        'event_level_matrix_flat': flattened_matrix_event_names,
        'global_scalar': scalar_global_keys,
        'global_vector_flat': flattened_vector_global_names,
        'global_matrix_flat': flattened_matrix_global_names,
        'all_flattened_features_ordered': all_flattened_feature_names_combined # Combined list for direct array indexing
    }

    # Iterate through each original event dictionary and prepare the new dictionary
    for event in translocation_events_data:
        prepared_event = {}

        # Copy original sequences and currents
        prepared_event['states'] = event.get('states')
        prepared_event['currents'] = event.get('currents')

        # Copy scalar event-level features
        for key in scalar_event_keys:
             prepared_event[key] = event.get(key)

        # Flatten and add vector event features with dynamic names
        for key in vector_event_keys:
            vector_data = event.get(key)
            if vector_data is not None and isinstance(vector_data, np.ndarray) and vector_data.shape == (vector_dim,):
                 flattened_vector = vector_data.flatten()
                 for i in range(vector_dim):
                      prepared_event[f'{key}_state_{i}'] = flattened_vector[i]
            else: # Add NaNs if missing or wrong shape data
                 for i in range(vector_dim):
                      prepared_event[f'{key}_state_{i}'] = np.nan

        # Flatten and add matrix event features with dynamic names
        for key in matrix_event_keys:
            matrix_data = event.get(key)
            if matrix_data is not None and isinstance(matrix_data, np.ndarray) and matrix_data.shape == matrix_shape:
                 flattened_matrix = matrix_data.flatten()
                 k = 0
                 for i in range(matrix_shape[0]):
                      for j in range(matrix_shape[1]):
                           prepared_event[f'{key}_state_{i}_state_{j}'] = flattened_matrix[k]
                           k += 1
            else: # Add NaNs
                 matrix_size = matrix_shape[0] * matrix_shape[1]
                 for i in range(matrix_size):
                      prepared_event[f'{key}_state_{int(i / matrix_shape[1])}_state_{i % matrix_shape[1]}'] = np.nan


        # Copy scalar global features
        for key in scalar_global_keys:
             prepared_event[key] = event.get(key)

        # Flatten and add vector global features with dynamic names
        for key in vector_global_keys:
            vector_data = event.get(key) # Global vector data is the same for all events after compute_global_features
            if vector_data is not None and isinstance(vector_data, np.ndarray) and vector_data.shape == (vector_dim,):
                 flattened_vector = vector_data.flatten()
                 for i in range(vector_dim):
                      prepared_event[f'{key}_state_{i}'] = flattened_vector[i]
            else: # Add NaNs
                 for i in range(vector_dim):
                      prepared_event[f'{key}_state_{i}'] = np.nan

        # Flatten and add matrix global features with dynamic names
        for key in matrix_global_keys:
            matrix_data = event.get(key) # Global matrix data is the same for all events
            if matrix_data is not None and isinstance(matrix_data, np.ndarray) and matrix_data.shape == matrix_shape:
                 flattened_matrix = matrix_data.flatten()
                 k = 0
                 for i in range(matrix_shape[0]):
                      for j in range(matrix_shape[1]):
                           prepared_event[f'{key}_state_{i}_state_{j}'] = flattened_matrix[k]
                           k += 1
            else: # Add NaNs
                 matrix_size = matrix_shape[0] * matrix_shape[1]
                 for i in range(matrix_size):
                      prepared_event[f'{key}_state_{int(i / matrix_shape[1])}_state_{i % matrix_shape[1]}'] = np.nan


        ml_dl_events_data.append(prepared_event)

    print(f"Prepared data for {len(ml_dl_events_data)} events with flattened features and original data.")
    print(f"Generated {len(all_flattened_feature_names_combined)} flattened feature names.")

    # Return the list of prepared dictionaries and the dictionary of categorized feature names lists
    return ml_dl_events_data, feature_names_dict
