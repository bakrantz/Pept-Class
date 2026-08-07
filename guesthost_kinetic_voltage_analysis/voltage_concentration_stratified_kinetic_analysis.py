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
    """
    if raw_states.size == 0:
        return [], [], 0

    open_state = np.max(raw_states) # The highest observed integer value state is assumed to be the open state.

    event_currents = []
    state_sequences = []

    open_state_indices = np.where(raw_states == open_state)[0]

    if open_state_indices.size < 2: 
        return [], [], open_state
    
    current_index = 0

    min_length_timepoints = 0
    if min_duration_ms is not None:
        min_length_timepoints = int(min_duration_ms * sampling_rate_hz / 1000.0)
        if min_length_timepoints == 0 and min_duration_ms > 0:
            min_length_timepoints = 1

    while current_index < len(raw_states):
        while current_index < len(raw_states) and raw_states[current_index] == open_state:
            current_index += 1

        if current_index < len(raw_states) and raw_states[current_index] != open_state:
            event_start_index = current_index
            
            search_end_index = event_start_index + 1
            while search_end_index < len(raw_states) and raw_states[search_end_index] != open_state:
                search_end_index += 1

            if search_end_index < len(raw_states) and raw_states[search_end_index] == open_state:
                event_end_index = search_end_index

                if event_start_index < event_end_index:
                    segmented_state_sequence = raw_states[event_start_index : event_end_index].tolist()
                    segmented_current_trace = scaled_raw_current[event_start_index : event_end_index]

                    if any(state != open_state for state in segmented_state_sequence):
                        event_length_timepoints = len(segmented_state_sequence)

                        if min_duration_ms is not None and event_length_timepoints < min_length_timepoints:
                            current_index = event_end_index 
                            continue 
                        else:
                            state_sequences.append(segmented_state_sequence)
                            event_currents.append(segmented_current_trace)
                    
                    current_index = event_end_index 
                else: 
                    current_index += 1 
            else: 
                break 
        else: 
            break 

    return event_currents, state_sequences, open_state

def _process_single_event(state_sequence, open_state):
    dwells_by_transition = {}
    current_state = state_sequence[0]
    current_dwell = 0

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
            
            dwells_by_transition[transition].append(current_dwell)
            
            current_state = to_state
            current_dwell = 1
            
    return dwells_by_transition

def get_all_dwells(list_of_state_sequences, open_state, time_sampling):
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
        
        for (from_state, to_state), dwells_in_points in dwells_dict.items():
            dwells_in_seconds = [d / time_sampling for d in dwells_in_points]
            dwell_matrix[from_state, to_state].extend(dwells_in_seconds)
            
    return dwell_matrix

def ln_survival_func(data):
    if not isinstance(data, (list, np.ndarray)) or len(data) == 0:
        return None, None
        
    sorted_data = np.sort(data)
    survival = 1 - (np.arange(1, len(sorted_data) + 1) / len(sorted_data))
    
    non_zero_survival_indices = survival > 0
    sorted_data_filtered = sorted_data[non_zero_survival_indices]
    survival_filtered = survival[non_zero_survival_indices]
    
    ln_survival_function = np.log(survival_filtered)
    
    return sorted_data_filtered, ln_survival_function

def one_exp_model_log(t, A, tau):
    return np.log(np.maximum(1e-10, A)) - t / tau

def get_ln_survival_and_fit_matrix_1exp(dwell_matrix):
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
            
            try:
                p0 = [1.0, 0.01] 
                bounds = ([1e-9, 1e-6], [1.1, 10.0])

                popt, pcov = curve_fit(one_exp_model_log, times, ln_survival, p0=p0, bounds=bounds)
                A, tau = popt
                perr = np.sqrt(np.diag(pcov)) 
                
                y_predicted = one_exp_model_log(times, A, tau)
                ss_total = np.sum((ln_survival - np.mean(ln_survival)) ** 2)
                ss_residual = np.sum((ln_survival - y_predicted) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 1.0

                n = len(times) 
                k = 2 
                bic = np.nan 

                if np.isfinite(ss_residual):
                    if ss_residual > 0:
                        bic = n * np.log(ss_residual / n) + k * np.log(n)
                    else:
                        bic = n * np.log(1e-12) + k * np.log(n)

                residuals = ln_survival - y_predicted
                
                results_matrix[i, j] = {
                    'times': times,
                    'ln_survival': ln_survival,
                    'fit': {
                        'A': A, 'tau': tau, 'r_squared': r_squared, 'bic': bic,
                        'A_err': perr[0], 'tau_err': perr[1],
                        'tau_mean': tau, 'tau_mean_err': perr[1] 
                    },
                    'residuals': residuals
                }
            except RuntimeError as e:
                results_matrix[i, j] = {
                    'times': times, 'ln_survival': ln_survival,
                    'fit': {
                        'A': np.nan, 'tau': np.nan, 'r_squared': np.nan, 'bic': np.nan,
                        'A_err': np.nan, 'tau_err': np.nan, 'tau_mean': np.nan, 'tau_mean_err': np.nan
                    },
                    'residuals': np.full_like(times, np.nan)
                }
                
    return results_matrix

def two_exp_model_log(t, A1, tau1, A2, tau2):
    return np.log(np.maximum(1e-10, A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)))

def get_ln_survival_and_fit_matrix_2exp(dwell_matrix):
    rows, cols = dwell_matrix.shape
    results_matrix = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            dwells = dwell_matrix[i, j]
            
            if dwells is None or len(dwells) < 5: 
                results_matrix[i, j] = None
                continue

            times, ln_survival = ln_survival_func(dwells)
            
            if times is None or len(times) < 5:
                results_matrix[i, j] = None
                continue
            try:
                p0 = [0.5, 0.005, 0.5, 0.05] 
                bounds = ([0.0, 1e-6, 0.0, 1e-6], [1.0, 1.0, 1.0, 1.0])
                
                popt, pcov = curve_fit(two_exp_model_log, times, ln_survival, p0=p0, bounds=bounds)
                perr = np.sqrt(np.diag(pcov)) 
                
                y_predicted = two_exp_model_log(times, *popt)

                A1, tau1, A2, tau2 = popt
                tau_mean = (A1 * tau1) + (A2 * tau2)
                J = np.array([tau1, A1, tau2, A2]) 
                tau_mean_var = J.T @ pcov @ J
                tau_mean_err = np.sqrt(tau_mean_var) if tau_mean_var > 0 else np.nan
                
                ss_total = np.sum((ln_survival - np.mean(ln_survival)) ** 2)
                ss_residual = np.sum((ln_survival - y_predicted) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 1.0

                n = len(times) 
                k = 4 
                bic = np.nan 

                if np.isfinite(ss_residual):
                    if ss_residual > 0:
                        bic = n * np.log(ss_residual / n) + k * np.log(n)
                    else:
                        bic = n * np.log(1e-12) + k * np.log(n)
                        
                residuals = ln_survival - y_predicted
                
                results_matrix[i, j] = {
                    'times': times, 'ln_survival': ln_survival,
                    'fit': {
                        'A1': popt[0], 'tau1': popt[1], 'A2': popt[2], 'tau2': popt[3],
                        'r_squared': r_squared, 'bic': bic,
                        'A1_err': perr[0], 'tau1_err': perr[1], 'A2_err': perr[2], 'tau2_err': perr[3],
                        'tau_mean': tau_mean, 'tau_mean_err': tau_mean_err
                    },
                    'residuals': residuals
                }

            except RuntimeError as e:
                results_matrix[i, j] = {
                    'times': times, 'ln_survival': ln_survival,
                    'fit': {
                        'A1': np.nan, 'tau1': np.nan, 'A2': np.nan, 'tau2': np.nan,
                        'r_squared': np.nan, 'bic': np.nan,
                        'A1_err': np.nan, 'tau1_err': np.nan, 'A2_err': np.nan, 'tau2_err': np.nan,
                        'tau_mean': np.nan, 'tau_mean_err': np.nan
                    },
                    'residuals': np.full_like(times, np.nan)
                }
    
    return results_matrix

def three_exp_model_log(t, A1, tau1, A2, tau2, A3, tau3):
    return np.log(np.maximum(1e-10, A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + A3 * np.exp(-t / tau3)))

def get_ln_survival_and_fit_matrix_3exp(dwell_matrix):
    rows, cols = dwell_matrix.shape
    results_matrix = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            dwells = dwell_matrix[i, j]
            
            if dwells is None or len(dwells) < 10:
                results_matrix[i, j] = None
                continue

            times, ln_survival = ln_survival_func(dwells)
            
            if times is None or len(times) < 10:
                results_matrix[i, j] = None
                continue
            
            try:
                p0 = [0.33, 0.005, 0.33, 0.05, 0.34, 0.5] 
                bounds = ([0.0, 1e-6, 0.0, 1e-6, 0.0, 1e-6], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                
                popt, pcov = curve_fit(three_exp_model_log, times, ln_survival, p0=p0, bounds=bounds)
                perr = np.sqrt(np.diag(pcov)) 
                
                y_predicted = three_exp_model_log(times, *popt)

                A1, tau1, A2, tau2, A3, tau3 = popt
                tau_mean = (A1 * tau1) + (A2 * tau2) + (A3 * tau3)
                J = np.array([tau1, A1, tau2, A2, tau3, A3])
                tau_mean_var = J.T @ pcov @ J
                tau_mean_err = np.sqrt(tau_mean_var) if tau_mean_var > 0 else np.nan
                
                ss_total = np.sum((ln_survival - np.mean(ln_survival)) ** 2)
                ss_residual = np.sum((ln_survival - y_predicted) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 1.0

                n = len(times) 
                k = 6 
                bic = np.nan 

                if np.isfinite(ss_residual):
                    if ss_residual > 0:
                        bic = n * np.log(ss_residual / n) + k * np.log(n)
                    else:
                        bic = n * np.log(1e-12) + k * np.log(n)
                
                residuals = ln_survival - y_predicted
                
                results_matrix[i, j] = {
                    'times': times, 'ln_survival': ln_survival,
                    'fit': {
                        'A1': popt[0], 'tau1': popt[1], 'A2': popt[2], 'tau2': popt[3], 'A3': popt[4], 'tau3': popt[5],
                        'r_squared': r_squared, 'bic': bic,
                        'A1_err': perr[0], 'tau1_err': perr[1], 'A2_err': perr[2], 'tau2_err': perr[3], 'A3_err': perr[4], 'tau3_err': perr[5],
                        'tau_mean': tau_mean, 'tau_mean_err': tau_mean_err
                    },
                    'residuals': residuals
                }

            except RuntimeError as e:
                results_matrix[i, j] = {
                    'times': times, 'ln_survival': ln_survival,
                    'fit': {
                        'A1': np.nan, 'tau1': np.nan, 'A2': np.nan, 'tau2': np.nan, 'A3': np.nan, 'tau3': np.nan,
                        'r_squared': np.nan, 'bic': np.nan,
                        'A1_err': np.nan, 'tau1_err': np.nan, 'A2_err': np.nan, 'tau2_err': np.nan, 'A3_err': np.nan, 'tau3_err': np.nan,
                        'tau_mean': np.nan, 'tau_mean_err': np.nan
                    },
                    'residuals': np.full_like(times, np.nan)
                }
    
    return results_matrix

def plot_ln_survival_and_residuals(results_matrix, nanopore_name, peptide_name, voltage, peptide_conc, model_type, plot_dir='./plots/'):
    """
    Generates and saves two pages of plots for a given model type.
    Now incorporates voltage and concentration into the title and filename.
    """
    os.makedirs(plot_dir, exist_ok=True)

    rows, cols = results_matrix.shape
    
    data_transitions = []
    for i in range(rows):
        for j in range(cols):
            if results_matrix[i, j] is not None:
                data_transitions.append((i, j, results_matrix[i, j]))

    if not data_transitions:
        print(f"No transitions with sufficient data to plot for {model_type} model.")
        return

    num_plots = len(data_transitions)
    grid_size = int(np.ceil(np.sqrt(num_plots)))
    fig_size = (5 * grid_size, 4 * grid_size)

    # --- Page 1: Log-Survival Plots ---
    fig1, axes1 = plt.subplots(grid_size, grid_size, figsize=fig_size)
    axes1 = np.array(axes1).flatten()
        
    for idx, (i, j, result) in enumerate(data_transitions):
        ax = axes1[idx]
        
        times = result['times']
        ln_survival = result['ln_survival']
        fit = result['fit']
        
        ax.plot(times, ln_survival, 'o', label='Data', markersize=4)
        
        if model_type == '1-exp' and not np.isnan(fit['A']):
            y_predicted = one_exp_model_log(times, fit['A'], fit['tau'])
            ax.plot(times, y_predicted, '-', color='r', linewidth=2, label='1-Exp Fit')
        
        elif model_type == '2-exp' and not np.isnan(fit['A1']):
            y_predicted = two_exp_model_log(times, fit['A1'], fit['tau1'], fit['A2'], fit['tau2'])
            ax.plot(times, y_predicted, '-', color='r', linewidth=2, label='2-Exp Fit')

        elif model_type == '3-exp' and not np.isnan(fit['A1']):
            y_predicted = three_exp_model_log(times, fit['A1'], fit['tau1'], fit['A2'], fit['tau2'], fit['A3'], fit['tau3'])
            ax.plot(times, y_predicted, '-', color='r', linewidth=2, label='3-Exp Fit')
        
        ax.set_title(f'State {i} $\\rightarrow$ State {j}', fontsize=10)
        ax.set_xlabel('Dwell Time ($s$)', fontsize=8)
        ax.set_ylabel('ln(Survival)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=8)

    for k in range(num_plots, len(axes1)):
        axes1[k].axis('off')
        
    fig1.suptitle(f'{model_type.upper()} Log-Survival Plots: {peptide_name} | {voltage}mV | {peptide_conc}nM', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    survival_file = f'{nanopore_name}_{peptide_name}_{voltage}mV_{peptide_conc}nM_{model_type}_log_survival_plots.png'
    fig1.savefig(os.path.join(plot_dir, survival_file))
    plt.close(fig1)

    # --- Page 2: Residuals Plots ---
    fig2, axes2 = plt.subplots(grid_size, grid_size, figsize=fig_size)
    axes2 = np.array(axes2).flatten()
        
    for idx, (i, j, result) in enumerate(data_transitions):
        ax = axes2[idx]

        times = result['times']
        residuals = result['residuals']
        
        if not np.all(np.isnan(residuals)):
            ax.plot(times, residuals, 'o', markersize=4)
            ax.axhline(0, color='r', linestyle='--')
        
        ax.set_title(f'State {i} $\\rightarrow$ State {j} Residuals', fontsize=10)
        ax.set_xlabel('Dwell Time ($s$)', fontsize=8)
        ax.set_ylabel('Residuals', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
    for k in range(num_plots, len(axes2)):
        axes2[k].axis('off')
        
    fig2.suptitle(f'Residuals of {model_type.upper()} Fits: {peptide_name} | {voltage}mV | {peptide_conc}nM', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    residual_file = f'{nanopore_name}_{peptide_name}_{voltage}mV_{peptide_conc}nM_{model_type}_residuals_plots.png'
    fig2.savefig(os.path.join(plot_dir, residual_file))
    plt.close(fig2)


def save_results_to_csv(results_matrix, nanopore_name, peptide_name, voltage, peptide_conc, model_type, output_filepath):
    """
    Saves the kinetic analysis results to a CSV file, properly tracking Voltage and Conc.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    results_list = []
    rows, cols = results_matrix.shape
    for i in range(rows):
        for j in range(cols):
            result = results_matrix[i, j]
            if result is not None:
                row_data = {
                    'nanopore_name': nanopore_name,
                    'peptide_name': peptide_name,
                    'voltage': voltage,
                    'peptide_conc': peptide_conc,
                    'from_state': i,
                    'to_state': j,
                    'r_squared': result['fit']['r_squared'],
                    'bic': result['fit']['bic']
                }
                
                if model_type == '1-exp':
                    row_data.update({
                        'A': result['fit']['A'], 'A_err': result['fit']['A_err'],
                        'tau': result['fit']['tau'], 'tau_err': result['fit']['tau_err'],
                        'tau_mean': result['fit']['tau_mean'], 'tau_mean_err': result['fit']['tau_mean_err']
                    })
                elif model_type == '2-exp':
                    row_data.update({
                        'A1': result['fit']['A1'], 'A1_err': result['fit']['A1_err'],
                        'tau1': result['fit']['tau1'], 'tau1_err': result['fit']['tau1_err'],
                        'A2': result['fit']['A2'], 'A2_err': result['fit']['A2_err'],
                        'tau2': result['fit']['tau2'], 'tau2_err': result['fit']['tau2_err'],
                        'tau_mean': result['fit']['tau_mean'], 'tau_mean_err': result['fit']['tau_mean_err']
                    })
                elif model_type == '3-exp':
                    row_data.update({
                        'A1': result['fit']['A1'], 'A1_err': result['fit']['A1_err'],
                        'tau1': result['fit']['tau1'], 'tau1_err': result['fit']['tau1_err'],
                        'A2': result['fit']['A2'], 'A2_err': result['fit']['A2_err'],
                        'tau2': result['fit']['tau2'], 'tau2_err': result['fit']['tau2_err'],
                        'A3': result['fit']['A3'], 'A3_err': result['fit']['A3_err'],
                        'tau3': result['fit']['tau3'], 'tau3_err': result['fit']['tau3_err'],
                        'tau_mean': result['fit']['tau_mean'], 'tau_mean_err': result['fit']['tau_mean_err']
                    })
                results_list.append(row_data)

    if not results_list:
        return

    df = pd.DataFrame(results_list)
    
    if os.path.exists(output_filepath):
        df.to_csv(output_filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(output_filepath, mode='w', header=True, index=False)

def consolidate_and_save_best_models(results_1exp, results_2exp, results_3exp, 
                                     nanopore_name, peptide_name, voltage, peptide_conc, output_dir='./results/'):
    master_results = []
    rows, cols = results_1exp.shape

    for i in range(rows):
        for j in range(cols):
            models = {'1-exp': results_1exp[i, j], '2-exp': results_2exp[i, j], '3-exp': results_3exp[i, j]}

            best_model_name = None
            lowest_bic = float('inf')
            
            for name, result in models.items():
                if result and 'bic' in result['fit'] and np.isfinite(result['fit']['bic']):
                    if result['fit']['bic'] < lowest_bic:
                        lowest_bic = result['fit']['bic']
                        best_model_name = name

            if best_model_name is None: continue

            best_result = models[best_model_name]
            fit = best_result['fit']

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
            
            components.sort(key=lambda x: x['tau'])

            consolidated_row = {
                'peptide_name': peptide_name,
                'nanopore_name': nanopore_name,
                'voltage': voltage,
                'peptide_conc': peptide_conc,
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

    if not master_results: return
        
    df = pd.DataFrame(master_results)
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f'{nanopore_name}_{peptide_name}_{voltage}mV_{peptide_conc}nM_consolidated_kinetics.csv')
    df.to_csv(output_filepath, index=False, float_format='%.6g')

def consolidate_and_save_best_models_dbic(results_1exp, results_2exp, results_3exp,
                                          nanopore_name, peptide_name, voltage, peptide_conc, output_dir='./results/',
                                          dbic_threshold=6):
    master_results = []
    rows, cols = results_1exp.shape

    for i in range(rows):
        for j in range(cols):
            models_data = {'1-exp': results_1exp[i, j], '2-exp': results_2exp[i, j], '3-exp': results_3exp[i, j]}

            bics = {name: res['fit']['bic'] for name, res in models_data.items() 
                    if res and 'bic' in res['fit'] and np.isfinite(res['fit']['bic'])}

            if not bics: continue

            best_model_name = '1-exp' if '1-exp' in bics else (list(bics.keys())[0])

            if '1-exp' in bics and '2-exp' in bics:
                if (bics['1-exp'] - bics['2-exp']) > dbic_threshold:
                    best_model_name = '2-exp'

            current_best_bic = bics[best_model_name]
            if '3-exp' in bics:
                if (current_best_bic - bics['3-exp']) > dbic_threshold:
                    best_model_name = '3-exp'

            best_result = models_data[best_model_name]
            fit = best_result['fit']

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
            
            components.sort(key=lambda x: x['tau'])

            consolidated_row = {
                'peptide_name': peptide_name,
                'nanopore_name': nanopore_name,
                'voltage': voltage,
                'peptide_conc': peptide_conc,
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

    if not master_results: return
        
    df = pd.DataFrame(master_results)
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f'{nanopore_name}_{peptide_name}_{voltage}mV_{peptide_conc}nM_consolidated_kinetics_dbic.csv')
    df.to_csv(output_filepath, index=False, float_format='%.6g')
    
if __name__ == '__main__':
    # 1. HARDCODED TARGET: Only analyzing these peptides
    peptide_names = ["guesthost_Ala", "guesthost_Leu", "guesthost_Phe", "guesthost_Thr", "guesthost_Trp", "guesthost_Tyr"]
    nanopore_name = 'PA'
    min_duration_ms = 5

    # Ensure the results directory exists before we try to save any pickle files
    os.makedirs('./results/', exist_ok=True)

    # --- Calculate Absolute Paths for Databases ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    database_dir = os.path.join(project_root, 'database')
    raw_db_json_path = os.path.join(database_dir, 'peptide_data.json')

    db = PeptideDatabase(db_file=raw_db_json_path)
    
    for peptide_name in peptide_names:
        print(f"\n--- Gathering all records for {peptide_name} ---")
        
        # 2. Broader Query: Remove hardcoded voltage and concentration
        base_query = {
            'experimental': True,
            'nanopore_name': nanopore_name,
            'peptide_name': peptide_name
        }
        records = db.retrieve_records(base_query)
        
        if not records:
            print(f"No records found for peptide '{peptide_name}'. Skipping.")
            continue
            
        # 3. Dynamic Grouping: Separate by Voltage and Concentration
        grouped_records = {}
        for r in records:
            key = (r.voltage, r.peptide_conc)
            if key not in grouped_records:
                grouped_records[key] = []
            grouped_records[key].append(r)
            
        print(f"Found {len(records)} total records, spanning {len(grouped_records)} unique Voltage/Concentration combinations.")
        
        # 4. Independent Processing Loop
        for (voltage, conc), group in grouped_records.items():
            print(f"\n===================================================================")
            print(f"Executing Kinetic Extraction: {voltage} mV | {conc} nM")
            print(f"Files in this cohort: {len(group)}")
            print(f"===================================================================")
            
            all_state_sequences = []
            time_sampling = None
            
            for r in group:
                data_file = r.data_file
                data_path = r.data_path
                filepath = os.path.join(data_path, data_file)
                time_sampling = r.time_sampling
                
                raw_times, raw_current, raw_states = load_stream(filepath)

                if raw_states.size > 0:
                    _, states, open_state = segment_translocations(raw_current, raw_states, sampling_rate_hz=time_sampling, min_duration_ms=min_duration_ms)
                    all_state_sequences.extend(states)
                else:
                    print(f"  -> Skipping record due to empty/invalid data: {data_file}")
                    
            if not all_state_sequences:
                print(f"  -> No valid translocation events found for {voltage}mV / {conc}nM. Skipping.")
                continue
                
            dwell_matrix = get_all_dwells(all_state_sequences, open_state, time_sampling)
            
            # --- 1-Exponential Model Analysis ---
            print("\n  [1] Performing single exponential fit...")
            results_matrix_1exp = get_ln_survival_and_fit_matrix_1exp(dwell_matrix)

            pkl_file_path = f'./results/{nanopore_name}_{peptide_name}_{voltage}mV_{conc}nM_1-exp_results.pkl'
            with open(pkl_file_path, 'wb') as file:
                pickle.dump(results_matrix_1exp, file)
            
            plot_ln_survival_and_residuals(results_matrix_1exp, nanopore_name, peptide_name, voltage, conc, model_type='1-exp')
            
            output_csv_path_1exp = f'./results/{nanopore_name}_{peptide_name}_{voltage}mV_{conc}nM_1-exp_results.csv'
            save_results_to_csv(results_matrix_1exp, nanopore_name, peptide_name, voltage, conc, model_type='1-exp', output_filepath=output_csv_path_1exp)
            
            # --- 2-Exponential Model Analysis ---
            print("  [2] Performing double exponential fit...")
            results_matrix_2exp = get_ln_survival_and_fit_matrix_2exp(dwell_matrix)

            pkl_file_path = f'./results/{nanopore_name}_{peptide_name}_{voltage}mV_{conc}nM_2-exp_results.pkl'
            with open(pkl_file_path, 'wb') as file:
                pickle.dump(results_matrix_2exp, file)
            
            plot_ln_survival_and_residuals(results_matrix_2exp, nanopore_name, peptide_name, voltage, conc, model_type='2-exp')
            
            output_csv_path_2exp = f'./results/{nanopore_name}_{peptide_name}_{voltage}mV_{conc}nM_2-exp_results.csv'
            save_results_to_csv(results_matrix_2exp, nanopore_name, peptide_name, voltage, conc, model_type='2-exp', output_filepath=output_csv_path_2exp)

            # --- 3-Exponential Model Analysis ---
            print("  [3] Performing triple exponential fit...")
            results_matrix_3exp = get_ln_survival_and_fit_matrix_3exp(dwell_matrix)

            pkl_file_path = f'./results/{nanopore_name}_{peptide_name}_{voltage}mV_{conc}nM_3-exp_results.pkl'
            with open(pkl_file_path, 'wb') as file:
                pickle.dump(results_matrix_3exp, file)
            
            plot_ln_survival_and_residuals(results_matrix_3exp, nanopore_name, peptide_name, voltage, conc, model_type='3-exp')
            
            output_csv_path_3exp = f'./results/{nanopore_name}_{peptide_name}_{voltage}mV_{conc}nM_3-exp_results.csv'
            save_results_to_csv(results_matrix_3exp, nanopore_name, peptide_name, voltage, conc, model_type='3-exp', output_filepath=output_csv_path_3exp)

            # Consolidate the best exponential models
            consolidate_and_save_best_models(
                results_matrix_1exp, results_matrix_2exp, results_matrix_3exp,
                nanopore_name, peptide_name, voltage, conc, output_dir='./results/'
            )

            consolidate_and_save_best_models_dbic(
                results_matrix_1exp, results_matrix_2exp, results_matrix_3exp,
                nanopore_name, peptide_name, voltage, conc, output_dir='./results/', dbic_threshold=6
            )
            print(f"  -> Successfully generated all arrays for {voltage}mV | {conc}nM.")
    
    print("\n--- All kinetic analysis stratifications are complete. ---")
