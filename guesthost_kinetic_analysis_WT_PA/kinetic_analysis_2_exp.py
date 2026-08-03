import pandas as pd
import numpy as np
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

def get_ln_survival_and_fit_matrix_1exp(dwell_matrix):
    """
    Processes a dwell time matrix to calculate the ln survival function for each
    transition and performs a linear fit (single exponential model).

    Args:
        dwell_matrix (numpy.ndarray): A matrix of lists containing dwell times in seconds.

    Returns:
        numpy.ndarray: A new matrix of dictionaries. Each dictionary contains:
                        - 'times': The sorted dwell times.
                        - 'ln_survival': The ln of the survival function.
                        - 'fit': A dictionary with 'slope', 'intercept', and 'r_squared'.
                        - 'residuals': The residuals of the linear fit.
                        - Returns None for transitions with no data.
    """
    rows, cols = dwell_matrix.shape
    results_matrix = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            dwells = dwell_matrix[i, j]
            
            if dwells is None or len(dwells) < 2:
                results_matrix[i, j] = None
                continue

            times, ln_survival = ln_survival_func(dwells)
            
            if times is None or len(times) < 2:
                results_matrix[i, j] = None
                continue

            # Perform a linear fit on the log-survival data.
            fit_coefficients = np.polyfit(times, ln_survival, 1)
            slope, intercept = fit_coefficients[0], fit_coefficients[1]
            
            # Calculate R-squared to quantify the quality of the fit
            y_predicted = slope * times + intercept
            ss_total = np.sum((ln_survival - np.mean(ln_survival)) ** 2)
            ss_residual = np.sum((ln_survival - y_predicted) ** 2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 1.0

            # Calculate the residuals
            residuals = ln_survival - y_predicted

            results_matrix[i, j] = {
                'times': times,
                'ln_survival': ln_survival,
                'fit': {'slope': slope, 'intercept': intercept, 'r_squared': r_squared},
                'residuals': residuals
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

    Args:
        dwell_matrix (numpy.ndarray): A matrix of lists containing dwell times in seconds.

    Returns:
        numpy.ndarray: A new matrix of dictionaries. Each dictionary contains:
                        - 'times': The sorted dwell times.
                        - 'ln_survival': The ln of the survival function.
                        - 'fit': A dictionary with 'A1', 'tau1', 'A2', 'tau2', and 'r_squared'.
                        - 'residuals': The residuals of the fit.
                        - Returns None for transitions with no data or if fit fails.
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
                # A1 and A2 are relative amplitudes, so they should sum to 1.
                # tau1 and tau2 are timescales.
                p0 = [0.5, 0.005, 0.5, 0.05] # A1, tau1, A2, tau2
                
                # Bounds for the parameters.
                # Amplitudes (A1, A2) must be positive.
                # Time constants (tau1, tau2) must be positive.
                bounds = ([0.0, 1e-6, 0.0, 1e-6], [1.0, 1.0, 1.0, 1.0])
                
                popt, pcov = curve_fit(two_exp_model_log, times, ln_survival, p0=p0, bounds=bounds)
                
                y_predicted = two_exp_model_log(times, *popt)
                
                # Calculate R-squared
                ss_total = np.sum((ln_survival - np.mean(ln_survival)) ** 2)
                ss_residual = np.sum((ln_survival - y_predicted) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 1.0
                
                # Calculate the residuals
                residuals = ln_survival - y_predicted
                
                results_matrix[i, j] = {
                    'times': times,
                    'ln_survival': ln_survival,
                    'fit': {'A1': popt[0], 'tau1': popt[1], 'A2': popt[2], 'tau2': popt[3], 'r_squared': r_squared},
                    'residuals': residuals
                }

            except RuntimeError as e:
                print(f"Curve fitting failed for transition {i} -> {j}: {e}")
                results_matrix[i, j] = None
    
    return results_matrix

def plot_ln_survival_and_residuals(results_matrix, nanopore_name, peptide_name, model_type, plot_dir='./plots/'):
    """
    Generates and saves two pages of plots for a given model type:
    1. Log-survival plots with model fits.
    2. Residuals of the fits.

    Args:
        results_matrix (numpy.ndarray): The matrix of dictionaries from the fitting function.
        nanopore_name (str): name of nanopore to label plots and save with descriptive filenames.
        peptide_name (str): name of peptide to label plots and save with descriptive filenames.
        model_type (str): Either '1-exp' or '2-exp' to determine plotting logic and titles.
        plot_dir (str): The directory to save the plots.
    """
    # Create the directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    rows, cols = results_matrix.shape
    
    # Filter out transitions with no data
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
    if num_plots > 1:
        axes1 = axes1.flatten()
    else:
        axes1 = [axes1]
        
    for idx, (i, j, result) in enumerate(data_transitions):
        ax = axes1[idx]
        
        times = result['times']
        ln_survival = result['ln_survival']
        fit = result['fit']
        
        # Plot the raw data points
        ax.plot(times, ln_survival, 'o', label='Data', markersize=4)
        
        # Plot the best-fit line or curve based on model_type
        if model_type == '1-exp':
            y_predicted = fit['slope'] * times + fit['intercept']
            ax.plot(times, y_predicted, '-', color='r', linewidth=2, label='Linear Fit')
        elif model_type == '2-exp':
            y_predicted = two_exp_model_log(times, fit['A1'], fit['tau1'], fit['A2'], fit['tau2'])
            ax.plot(times, y_predicted, '-', color='r', linewidth=2, label='2-Exp Fit')
        
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
    print(f"{model_type.upper()} log-survival plots saved to '{plot_dir}{survival_file}'")

    # --- Page 2: Residuals Plots ---
    fig2, axes2 = plt.subplots(grid_size, grid_size, figsize=fig_size)
    if num_plots > 1:
        axes2 = axes2.flatten()
    else:
        axes2 = [axes2]
        
    for idx, (i, j, result) in enumerate(data_transitions):
        ax = axes2[idx]

        times = result['times']
        residuals = result['residuals']

        # Plot the residuals
        ax.plot(times, residuals, 'o', markersize=4)
        ax.axhline(0, color='r', linestyle='--')
        
        ax.set_title(f'State {i} $\\rightarrow$ State {j} Residuals', fontsize=10)
        ax.set_xlabel('Dwell Time ($s$)', fontsize=8)
        ax.set_ylabel('Residuals', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
    # Hide any unused subplots
    for k in range(num_plots, len(axes2)):
        axes2[k].axis('off')
        
    fig2.suptitle(f'Residuals of {model_type.upper()} Fits', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    residual_file = f'{nanopore_name}_{peptide_name}_{model_type}_residuals_plots.png'
    fig2.savefig(os.path.join(plot_dir, residual_file))
    print(f"Residuals plots saved to '{plot_dir}{residual_file}'")
    
def save_results_to_csv(results_matrix, nanopore_name, peptide_name, model_type, output_filepath):
    """
    Saves the kinetic analysis results to a CSV file.

    Args:
        results_matrix (numpy.ndarray): Matrix of dictionaries from fitting function.
        nanopore_name (str): The name of the nanopore.
        peptide_name (str): The name of the peptide.
        model_type (str): Either '1-exp' or '2-exp' to determine which results to save.
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
                }
                if model_type == '1-exp':
                    row_data.update({
                        'slope': result['fit']['slope'],
                        'intercept': result['fit']['intercept'],
                        'r_squared': result['fit']['r_squared']
                    })
                elif model_type == '2-exp':
                    row_data.update({
                        'A1': result['fit']['A1'],
                        'tau1': result['fit']['tau1'],
                        'A2': result['fit']['A2'],
                        'tau2': result['fit']['tau2'],
                        'r_squared': result['fit']['r_squared']
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
        
        # Plot and save results for the single exponential model
        plot_ln_survival_and_residuals(results_matrix_1exp, nanopore_name, peptide_name, model_type='1-exp')
        
        # Save the kinetic analysis results to a separate CSV file
        output_csv_path_1exp = f'./results/{nanopore_name}_{peptide_name}_1-exp_results.csv'
        save_results_to_csv(results_matrix_1exp, nanopore_name, peptide_name, model_type='1-exp', output_filepath=output_csv_path_1exp)
        
        # --- 2-Exponential Model Analysis ---
        print("\nPerforming double exponential fit...")
        results_matrix_2exp = get_ln_survival_and_fit_matrix_2exp(dwell_matrix)
        
        # Plot and save results for the double exponential model
        plot_ln_survival_and_residuals(results_matrix_2exp, nanopore_name, peptide_name, model_type='2-exp')
        
        # Save the kinetic analysis results to a separate CSV file
        output_csv_path_2exp = f'./results/{nanopore_name}_{peptide_name}_2-exp_results.csv'
        save_results_to_csv(results_matrix_2exp, nanopore_name, peptide_name, model_type='2-exp', output_filepath=output_csv_path_2exp)

    print("\n--- All kinetic analysis and CSV saving complete. ---")
