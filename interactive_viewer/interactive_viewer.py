## Basic script to view ATF and CSV current vs. time records using Matplotlib
# Highlight snippet of data shift-click (but not while in magnifying glass mode of Matplotlib)
# Hit 's' on highlighted data and saves as an enumerated snippet
# Hit 'm' on highlighted data to get mean value

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# --- 1. User's Loading Functions (Unchanged) ---
def load_atf_stream(filepath: str, header_row_index: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at '{filepath}'")
    with open(filepath, 'r') as f:
        all_lines = f.readlines()
    if len(all_lines) < header_row_index + 2:
        raise ValueError(f"File '{filepath}' is too short.")
    header_lines = [line.strip('\n') for line in all_lines[:header_row_index + 1]]
    df = pd.read_csv(filepath, sep='\t', skiprows=header_row_index)
    df.columns = df.columns.str.strip().str.replace(' #', '').str.replace(' ', '_').str.replace('[()]', '', regex=True)
    
    required_cols = {"Time_s": "Time (s)", "Trace1_pA": "Trace #1 (pA)", "Trace1_mV": "Trace #1 (mV)"}
    for cleaned_name, original_name in required_cols.items():
        if cleaned_name not in df.columns:
            raise KeyError(f"Expected column '{original_name}' not found. Available: {df.columns.tolist()}")
    
    times = df["Time_s"].to_numpy()
    current = df["Trace1_pA"].to_numpy()
    voltage = df["Trace1_mV"].to_numpy()
    return times, current, voltage, header_lines

def load_csv_stream(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)
        if not all(col in df.columns for col in ['Time', 'Current', 'State']):
            raise ValueError("CSV must contain 'Time', 'Current', and 'State' columns.")
        return df['Time'].values, df['Current'].values, df['State'].values
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return np.array([]), np.array([]), np.array([])

# --- Configuration Setting ---
ZERO_TIME_ON_SAVE = True

# --- 2. Global variables ---
main_df = None
selected_snippet_df = None
source_file_type = None
atf_header_lines = None
snippet_counter = 1
time_column_name = None
current_column_name = None

# --- 3. Callback and Helper Functions ---

# <<< MODIFIED SECTION >>>
def onselect(xmin, xmax):
    """Callback for SpanSelector, now updates the plot title."""
    global selected_snippet_df
    indmin, indmax = np.searchsorted(main_df[time_column_name], (xmin, xmax))
    indmax = min(len(main_df) - 1, indmax)
    
    if indmin >= indmax: 
        ax.set_title("Interactive Data Selector") # Reset title on tiny selection
        fig.canvas.draw_idle()
        return

    selected_snippet_df = main_df.iloc[indmin:indmax]
    
    # Your excellent addition to the print statement
    delta_time = xmax - xmin
    print(f"Selected region from Time {xmin:.3f}s to {xmax:.3f}s of Delta_time {delta_time:.3f}s ({len(selected_snippet_df)} points).")
    print("-> Press 's' to save this snippet.")
    
    # Update the plot title with the delta_time
    ax.set_title(f"Interactive Data Selector | Selected Δt: {delta_time:.3f}s")
    fig.canvas.draw_idle() # Tell matplotlib to redraw the figure to show the new title

def save_current_snippet():
    """Saves the currently selected snippet."""
    global snippet_counter
    if selected_snippet_df is None or selected_snippet_df.empty:
        print("No snippet selected to save.")
        return

    snippet_to_save = selected_snippet_df
    if ZERO_TIME_ON_SAVE:
        snippet_to_save = selected_snippet_df.copy()
        first_time_point = snippet_to_save[time_column_name].iloc[0]
        snippet_to_save[time_column_name] = snippet_to_save[time_column_name] - first_time_point

    base_name, _ = os.path.splitext(FILE_TO_LOAD)
    output_filename = f"{base_name}_snippet_{snippet_counter}.{source_file_type}"

    if source_file_type == 'atf':
        with open(output_filename, 'w') as f:
            for line in atf_header_lines:
                f.write(line + '\n')
        snippet_to_save.to_csv(output_filename, mode='a', sep='\t', header=False, index=False, float_format='%.6f')
        print(f"✅ ATF snippet saved to: {output_filename}")

    elif source_file_type == 'csv':
        snippet_to_save.to_csv(output_filename, index=False, float_format='%.6f')
        print(f"✅ CSV snippet saved to: {output_filename}")

    snippet_counter += 1

def get_mean_current_of_snippet():
    """
    Calculates the mean current and updates the plot title with the result.
    """
    if selected_snippet_df is None or selected_snippet_df.empty:
        print("No snippet selected to compute mean.")
        return None

    current_series = selected_snippet_df[current_column_name]
    mean_current = current_series.mean()
    
    print(f'Mean current of selection is: {mean_current:.3f} pA')

    # --- Optional Addition: Update the plot title ---
    # Re-calculate delta_t from the snippet to ensure it's in the title
    time_values = selected_snippet_df[time_column_name].values
    delta_t = time_values[-1] - time_values[0]
    
    # Construct the new title with all info
    new_title = (f"Interactive Data Selector | Selected Δt: {delta_t:.3f}s "
                 f"| Mean I: {mean_current:.3f} pA")
    ax.set_title(new_title)
    fig.canvas.draw_idle() # Redraw the canvas to show the new title
    
    return mean_current
    
def on_key_press(event):
    if event.key == 's':
        save_current_snippet()
    elif event.key == 'm':
        get_mean_current_of_snippet()

# --- 4. Main Execution Logic ---
if __name__ == "__main__":
    FILE_TO_LOAD = "./data/11622000-guesthost_Phe-70_mV-400_Hz.atf" 

    if not os.path.exists(FILE_TO_LOAD):
        raise FileNotFoundError(f"Data file not found. Please update the FILE_TO_LOAD variable.")

    _, file_ext = os.path.splitext(FILE_TO_LOAD)
    file_ext = file_ext.lower()

    if file_ext == '.atf':
        print(f"Loading ATF file: {FILE_TO_LOAD}")
        source_file_type = 'atf'
        times, currents, voltages, atf_header_lines = load_atf_stream(FILE_TO_LOAD)
        main_df = pd.DataFrame({'Time': times, 'Current': currents, 'Voltage': voltages})
        main_df.columns = ['Time (s)', 'Trace #1 (pA)', 'Trace #1 (mV)']
        time_column_name = 'Time (s)'
        current_column_name = 'Trace #1 (pA)'

    elif file_ext == '.csv':
        print(f"Loading CSV file: {FILE_TO_LOAD}")
        source_file_type = 'csv'
        times, currents, states = load_csv_stream(FILE_TO_LOAD)
        if len(times) == 0:
            raise SystemExit("Failed to load CSV data. Exiting.")
        main_df = pd.DataFrame({'Time': times, 'Current': currents, 'State': states})
        time_column_name = 'Time'
        current_column_name = 'Current'
    
    else:
        raise ValueError(f"Unsupported file type: '{file_ext}'. Please use .atf or .csv files.")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(main_df[time_column_name], main_df[current_column_name], lw=0.5)
    
    ax.set_title("Interactive Data Selector") # Initial title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Current (pA)')
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    span = SpanSelector(
        ax, onselect, 'horizontal', useblit=True,
        props=dict(alpha=0.3, facecolor='lightgreen'),
        interactive=True, drag_from_anywhere=True
    )

    plt.show()
