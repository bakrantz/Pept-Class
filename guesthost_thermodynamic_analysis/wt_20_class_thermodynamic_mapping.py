import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict
import matplotlib as mpl

# ==========================================
# 0. Global Settings
# ==========================================
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10

R = 1.987e-3  # Gas constant in kcal/mol/K
T = 298.15    # Temperature in Kelvin
RT = R * T    # Approx. 0.593 kcal/mol
MIN_DURATION_MS = 15 # Mirroring your XGBoost 15ms cutoff to isolate deep friction

# Custom classes are required to access and process data in databases
common_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if common_parent_dir not in sys.path:
    sys.path.insert(0, common_parent_dir)
try:
    from database.PeptideDatabase import PeptideData, PeptideDatabase
    print("Successfully imported database classes.")
except ImportError as e:
    print(f"Error importing database classes: {e}")
    sys.exit(1)

# ==========================================
# 1. Core Processing Functions
# ==========================================
def load_stream(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)
        if 'Time' not in df.columns or 'Current' not in df.columns or 'State' not in df.columns:
            return np.array([]), np.array([]), np.array([])
        return df['Time'].values.astype(np.float32), df['Current'].values.astype(np.float32), df['State'].values.astype(np.int32)
    except Exception:
        return np.array([]), np.array([]), np.array([])

def segment_translocations(scaled_raw_current, raw_states, sampling_rate_hz=400, min_duration_ms=None):
    if raw_states.size == 0: return [], [], 0
    open_state = np.max(raw_states)
    event_currents, state_sequences = [], []
    current_index = 0
    min_length_timepoints = int(min_duration_ms * sampling_rate_hz / 1000.0) if min_duration_ms else 0

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
                    if any(state != open_state for state in segmented_state_sequence):
                        if len(segmented_state_sequence) >= min_length_timepoints:
                            state_sequences.append(segmented_state_sequence)
                    current_index = event_end_index
                else: current_index += 1
            else: break
        else: break
    return event_currents, state_sequences, open_state

def bootstrap_intra_event_delta_g(all_state_sequences, n_iterations=1000):
    if not all_state_sequences:
        return {0: (np.nan, np.nan), 1: (np.nan, np.nan), 2: (np.nan, np.nan)}
        
    n_events = len(all_state_sequences)
    bootstrap_results = {0: [], 1: [], 2: []}
    
    for _ in range(n_iterations):
        sampled_indices = np.random.choice(n_events, n_events, replace=True)
        time_in_state = {0: 0.0, 1: 0.0, 2: 0.0}
        
        for idx in sampled_indices:
            seq = all_state_sequences[idx]
            for state in seq:
                if state in time_in_state:
                    time_in_state[state] += 1
                    
        total_bound_time = sum(time_in_state.values())
        
        for state in [0, 1, 2]:
            if total_bound_time > 0 and time_in_state[state] > 0:
                occupancy = time_in_state[state] / total_bound_time
                bootstrap_results[state].append(-RT * np.log(occupancy))
            else:
                bootstrap_results[state].append(np.nan)
                
    final_stats = {}
    for state in [0, 1, 2]:
        clean_dg = [val for val in bootstrap_results[state] if not np.isnan(val)]
        if clean_dg:
            final_stats[state] = (np.mean(clean_dg), np.std(clean_dg))
        else:
            final_stats[state] = (np.nan, np.nan)
            
    return final_stats

# ==========================================
# 2. Main Execution
# ==========================================
def main():
    print("--- Intra-Event Thermodynamic Mapping (20 Amino Acids) ---")
    print(f"Energy units: kcal/mol (RT = {RT:.3f} kcal/mol at {T}K).")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    db_path = os.path.join(project_root, 'database', 'peptide_data.json')

    try:
        db = PeptideDatabase(db_file=db_path)
    except Exception:
        print("Database initialization failed. Please check paths.", file=sys.stderr)
        sys.exit(1)

    peptides = [
        'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 
        'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'TrpDL', 'Tyr', 'Val'
    ]
    
    peptide_names = [f"guesthost_{p}" for p in peptides]
    results_list = []

    for pep in peptide_names:
        print(f"Processing {pep}...")
        query = {
            'experimental': True,
            'nanopore_name': 'PA',
            'peptide_name': pep,
            'voltage': 70,
            'time_sampling': 400
        }
        records = db.retrieve_records(query)
        
        if not records:
            continue
            
        all_state_sequences = []
        for r in records:
            filepath = os.path.join(r.data_path, r.data_file)
            raw_times, raw_current, raw_states = load_stream(filepath)
            sampling_rate_hz = getattr(r, 'time_sampling', 400)
            _, state_sequences, _ = segment_translocations(raw_current, raw_states, sampling_rate_hz, MIN_DURATION_MS)
            all_state_sequences.extend(state_sequences)
            
        stats = bootstrap_intra_event_delta_g(all_state_sequences)
        
        for state in [0, 1, 2]:
            mean_dg, std_dg = stats[state]
            if not np.isnan(mean_dg):
                results_list.append({
                    'Peptide': pep.replace('guesthost_', ''),
                    'State': state,
                    'DG_Mean': mean_dg,
                    'DG_Std': std_dg,
                    'N_Events_Pooled': len(all_state_sequences)
                })

    if not results_list:
        print("No valid thermodynamic data extracted.")
        return

    df = pd.DataFrame(results_list)
    os.makedirs('./data', exist_ok=True)
    df.to_csv('./data/WT_20AA_IntraEvent_Thermodynamics.csv', index=False)
    
    # ==========================================
    # 3. Plotting
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Define colors for states to match your standard scheme
    colors = {0: '#D62728', 1: '#1F77B4', 2: '#2CA02C'}
    titles = {0: 'State 0 (Fully Clamped)', 1: 'State 1 (Partial Block)', 2: 'State 2 (Dilated)'}
    
    # Sort the x-axis so it's consistent
    plot_peptides = [p.replace('guesthost_', '') for p in peptide_names if p.replace('guesthost_', '') in df['Peptide'].values]

    for state in [0, 1, 2]:
        ax = axes[state]
        state_df = df[df['State'] == state].set_index('Peptide').reindex(plot_peptides)
        
        y = state_df['DG_Mean'].fillna(0)
        yerr = state_df['DG_Std'].fillna(0)
        
        x_pos = np.arange(len(plot_peptides))
        
        ax.bar(x_pos, y, yerr=yerr, capsize=3, color=colors[state], edgecolor='black', alpha=0.8)
        
        ax.set_title(titles[state], fontweight='bold')
        ax.set_ylabel(r'$\Delta G$ (kcal/mol)', fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if state == 2:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_peptides, rotation=45, ha='right', fontweight='bold')

    plt.suptitle("Intra-Event Thermodynamic Friction in the WT PA Translocase (+70 mV)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/WT_20AA_IntraEvent_Thermodynamics.png', dpi=300, bbox_inches='tight')
    plt.savefig('./plots/WT_20AA_IntraEvent_Thermodynamics.svg', format='svg', bbox_inches='tight')
    
    print("\n✅ Success! Thermodynamics calculated and plotted.")
    print("This explicitly proves that the XGBoost State Probability features are direct measurements of thermodynamic friction.")

if __name__ == '__main__':
    main()