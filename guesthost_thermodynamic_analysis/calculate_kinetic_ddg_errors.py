import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict

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
    # Do not exit here to allow the mock execution below, but warn the user.

# --- Configuration ---
R = 1.987e-3  # Gas constant in kcal/mol/K (User requested unit)
T = 298.15    # Temperature in Kelvin
RT = R * T    # Approx. 0.593 kcal/mol
DATA_DIR = '../database/' # Assuming data files are accessible relative to project root
MIN_DURATION_MS = 5 # Standard filter value for this analysis

# --- Placeholder/Required Classes and Functions ---

# NOTE: MockRecord is now only used for the final list structure, as we retrieve real PeptideData records.
class MockRecord:
    """Mock structure matching the retrieved database record."""
    def __init__(self, peptide_name, nanopore_name, data_file, data_path, time_sampling=400):
        self.peptide_name = peptide_name
        self.nanopore_name = nanopore_name
        self.data_file = data_file
        self.data_path = data_path
        self.time_sampling = time_sampling

def load_stream(csv_filepath):
    """Loads a CSV file containing raw translocation event data, extracts and scales
    the current, and extracts the state labels. (Returns empty arrays on error/empty)"""
    try:
        # NOTE: The database is assumed to store the FULL PATH or a path relative to DATA_PATH
        # For full path safety, we might need os.path.abspath(csv_filepath) here.
        df = pd.read_csv(csv_filepath)
        if 'Time' not in df.columns or 'Current' not in df.columns or 'State' not in df.columns:
            return np.array([]), np.array([]), np.array([])
        raw_times = df['Time'].values.astype(np.float32)
        raw_current = df['Current'].values.astype(np.float32)
        raw_states = df['State'].values.astype(np.int32)
        return raw_times, raw_current, raw_states
    except Exception as e:
        # print(f"Error loading stream {csv_filepath}: {e}", file=sys.stderr) # Suppress for clean run
        return np.array([]), np.array([]), np.array([])

def segment_translocations(scaled_raw_current, raw_states, sampling_rate_hz=1000, min_duration_ms=None):
    """Segments the raw state sequence and scaled current trace into individual translocation events."""
    if raw_states.size == 0: return [], [], 0
    open_state = np.max(raw_states)
    event_currents = []
    state_sequences = []
    current_index = 0
    min_length_timepoints = 0
    if min_duration_ms is not None: min_length_timepoints = int(min_duration_ms * sampling_rate_hz / 1000.0)

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
            else: break
        else: break
    
    return event_currents, state_sequences, open_state

def calculate_occupancy_delta_g_per_record(record):
    """
    Calculates the Delta G (occupancy) for States 0, 1, and 2 for a single record (replicate).
    Returns: dict {state: delta_g_kcal}
    """
    global RT
    
    # Construct filepath: assuming record.data_path and record.data_file form the path
    filepath = os.path.join(record.data_path, record.data_file)
    
    raw_times, raw_current, raw_states = load_stream(filepath)
    
    # Assuming standard parameters for segmentation
    sampling_rate_hz = record.time_sampling if hasattr(record, 'time_sampling') else 400
    min_duration_ms = MIN_DURATION_MS
    
    _, state_sequences, open_state = segment_translocations(raw_current, raw_states, sampling_rate_hz, min_duration_ms)
    
    if not state_sequences:
        return {0: np.nan, 1: np.nan, 2: np.nan}
        
    # Calculate total time in each bound state (0, 1, 2)
    total_time_in_state = {0: 0.0, 1: 0.0, 2: 0.0}
    
    for seq in state_sequences:
        current_state = seq[0]
        dwell_time_points = 0
        
        for state in seq:
            if state == open_state: break
            if state == current_state:
                dwell_time_points += 1
            else:
                total_time_in_state[current_state] += dwell_time_points
                current_state = state
                dwell_time_points = 1
        
        if current_state in total_time_in_state:
             total_time_in_state[current_state] += dwell_time_points

    total_bound_time_points = sum(total_time_in_state.values())
    
    # Calculate fractional occupancy and free energy
    delta_g_results = {}
    if total_bound_time_points == 0:
        return {0: np.nan, 1: np.nan, 2: np.nan}

    for state in [0, 1, 2]:
        time_in_state = total_time_in_state.get(state, 0)
        occupancy = time_in_state / total_bound_time_points
        
        delta_g = -RT * np.log(occupancy) if occupancy > 0 else np.nan
        delta_g_results[state] = delta_g
        
    return delta_g_results


def calculate_thermodynamics_with_errors(nanopore_records, ref_pore_records):
    """
    Calculates the mean and standard deviation of Delta Delta G (occupancy)
    using individual records (streams) as replicates.
    """
    
    # --- 1. Process WT (Reference) Data ---
    wt_g_per_record = defaultdict(lambda: {0: [], 1: [], 2: []})
    for record in ref_pore_records:
        peptide = record.peptide_name
        g_values = calculate_occupancy_delta_g_per_record(record)
        for state in [0, 1, 2]:
            if not np.isnan(g_values[state]): wt_g_per_record[peptide][state].append(g_values[state])
    
    # Calculate the mean WT G for each state (this is the non-replicable reference point)
    wt_g_mean = {}
    for peptide, state_data in wt_g_per_record.items():
        wt_g_mean[peptide] = {}
        for state in [0, 1, 2]:
            wt_g_mean[peptide][state] = np.mean(state_data[state]) if state_data[state] else np.nan
            
    # --- 2. Process Mutant Data (Calculate DDG and STD) ---
    mut_ddg_all = []
    
    # Dictionary to ensure we only process records belonging to the mutant pore (F427A or F427Y)
    mut_pore_name = nanopore_records[0].nanopore_name if nanopore_records else 'N/A'

    for record in nanopore_records:
        peptide = record.peptide_name
        g_values_mut = calculate_occupancy_delta_g_per_record(record)
        
        for state in [0, 1, 2]:
            wt_mean = wt_g_mean.get(peptide, {}).get(state)
            if not np.isnan(g_values_mut[state]) and not np.isnan(wt_mean):
                ddg_val = g_values_mut[state] - wt_mean
                mut_ddg_all.append({
                    'peptide': peptide,
                    'state': state,
                    'ddg': ddg_val
                })
    
    df_ddg = pd.DataFrame(mut_ddg_all)
    
    # Group by peptide and state to calculate mean and std. dev.
    summary_df = df_ddg.groupby(['peptide', 'state'])['ddg'].agg(
        ddg_mean_kcal=('mean'),
        ddg_std_kcal=('std'),
        num_replicates=('count')
    ).reset_index()

    summary_df['pore_variant'] = mut_pore_name
    
    return summary_df[['pore_variant', 'peptide', 'state', 'ddg_mean_kcal', 'ddg_std_kcal', 'num_replicates']]

# --- Plotting Function (No change needed) ---
def plot_thermodynamic_differences(df, peptides_to_plot):
    """
    Generates a 3-panel figure plotting the delta_delta_g for the mutants for each state,
    including error bars (STD DEV).
    """
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 8

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3), sharey=True)
    axes = np.array(axes).flatten()

    mutant_order = ['PA_F427Y', 'PA_F427A']
    legend_labels = {'PA_F427A': 'F427A', 'PA_F427Y': 'F427Y'}
    colors = {'PA_F427A': '#d62728', 'PA_F427Y': '#ff7f0e'}

    for i, state in enumerate([0, 1, 2]):
        ax = axes[i]
        
        # Prepare data for plotting
        plot_df = df[
            (df['state'] == state) &
            (df['peptide'].isin(peptides_to_plot)) &
            (df['pore_variant'].isin(mutant_order))
        ].copy()

        if plot_df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(f'State {state} Stability')
            continue

        # Pivot the data for easy bar plotting (Mean)
        pivot_mean = plot_df.pivot(index='peptide', columns='pore_variant', values='ddg_mean_kcal').reindex(peptides_to_plot)
        # Pivot the data for error bars (STD)
        pivot_std = plot_df.pivot(index='peptide', columns='pore_variant', values='ddg_std_kcal').reindex(peptides_to_plot)
        
        x = np.arange(len(peptides_to_plot))
        width = 0.35

        # F427Y Bars and Errors (Left)
        y_y = pivot_mean.get('PA_F427Y').fillna(0)
        e_y = pivot_std.get('PA_F427Y').fillna(0)
        ax.bar(x - width/2, y_y, width, yerr=e_y, capsize=3, label=legend_labels['PA_F427Y'], color=colors['PA_F427Y'])
        
        # F427A Bars and Errors (Right)
        y_a = pivot_mean.get('PA_F427A').fillna(0)
        e_a = pivot_std.get('PA_F427A').fillna(0)
        ax.bar(x + width/2, y_a, width, yerr=e_a, capsize=3, label=legend_labels['PA_F427A'], color=colors['PA_F427A'])

        if i == 0:
            ax.set_ylabel('$\Delta\Delta$G (kcal/mol)')
        
        ax.set_title(f'State {state} Stability')
        ax.set_xticks(x)
        ax.set_xticklabels(peptides_to_plot, rotation=45, ha="right")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)

    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate handles/labels before placing the legend
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(l)] for l in unique_labels]
    
    fig.legend(unique_handles, unique_labels, loc='upper right', title="Pore Variant")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    output_filename = 'Fig_thermodynamic_stability_comparison_with_errors.svg'
    plt.savefig(output_filename)
    print(f"Thermodynamic stability figure saved to '{output_filename}'")


if __name__ == '__main__':
    
    print("--- DDG (Occupancy) Error Analysis Script ---")
    print(f"Energy units set to kcal/mol (RT = {RT:.3f} kcal/mol).")

    # --- Calculate Absolute Paths for Databases ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    database_dir = os.path.join(project_root, 'database')
    raw_db_json_path = os.path.join(database_dir, 'peptide_data.json')

    # Ensure PeptideDatabase is initialized
    try:
        db = PeptideDatabase(db_file=raw_db_json_path)
    except NameError:
        # If the import failed silently due to NameError (e.g., if NameError was caught above)
        print("Database initialization failed. Please check the import path.", file=sys.stderr)
        sys.exit(1)


    all_records = []
    peptides = ["guesthost_Ala", "guesthost_Leu", "guesthost_Phe", "guesthost_Thr", "guesthost_Trp", "guesthost_TrpDL", "guesthost_Tyr"]
    pore_variants = ['PA', 'PA_F427A', 'PA_F427Y']
    
    for pore_name in pore_variants:
        for peptide in peptides:
            print(f"\n--- Accessing data for {peptide} via {pore_name} pore ---")
            peptide_query = {
                'experimental': True,
                'nanopore_name': pore_name,
                'peptide_name': peptide, # CORRECTED: Using current loop variable 'peptide'
                'voltage': 70,
                'time_sampling': 400,
                'peptide_conc': {'$gte': 5, '$lte': 20}
            }
            result_peptide_records = db.retrieve_records(peptide_query)
            
            # Append retrieved records (which are assumed to be the replicates/streams)
            for r in result_peptide_records:
                # The retrieved records (r) are already PeptideData objects, so we just append them.
                all_records.append(r)
                
    # Check if any data was retrieved
    if not all_records:
        print("\nERROR: No records retrieved from database. Cannot proceed with analysis.", file=sys.stderr)
        sys.exit(1)
        
    # Separate records into Mutant and Reference (WT)
    wt_records = [r for r in all_records if r.nanopore_name == 'PA']
    f427a_records = [r for r in all_records if r.nanopore_name == 'PA_F427A']
    f427y_records = [r for r in all_records if r.nanopore_name == 'PA_F427Y']


    # --- Perform Calculations ---
    
    print("\n--- Running DDG Calculations (F427A vs WT) ---")
    f427a_summary = calculate_thermodynamics_with_errors(f427a_records, wt_records)

    print("\n--- Running DDG Calculations (F427Y vs WT) ---")
    f427y_summary = calculate_thermodynamics_with_errors(f427y_records, wt_records)

    # --- Plotting and Final CSV Generation ---
    
    final_thermo_df = pd.concat([f427a_summary, f427y_summary])
    
    # Save the final data set (needed for Figure 2 and subsequent analysis)
    output_csv_path = './data/thermodynamic_stability_summary_with_errors.csv'
    final_thermo_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    print(f"\nThermodynamic stability summary saved to '{output_csv_path}'")

    # Generate the Figure 2 equivalent plot
    plot_thermodynamic_differences(final_thermo_df, peptides)
    
    print("\nScript completed successfully.")
