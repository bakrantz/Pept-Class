import numpy as np
import pandas as pd
import os
import shutil
from scipy import signal
import matplotlib.pyplot as plt
import math
import re
import datetime

# --- Step 1: Function to Load ATF data (Header & Numerical Data) ---
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
    with open(filepath, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    if len(all_lines) < header_row_index + 2: # At least header line + 1 data row
        raise ValueError(f"File '{filepath}' is too short to contain header and data as expected.")

    # Extract header lines (from ATF 1.0 to column names line, inclusive)
    # The header_row_index is the line number of the actual column names
    # So we take lines from 0 up to and including header_row_index
    header_lines = [line.strip('\n') for line in all_lines[:header_row_index + 1]]

    # Read data using pandas, skipping header lines
    # skiprows is now used for the actual number of lines to skip before data
    # (i.e., skipping up to and including the header_row_index line)
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

# --- Step 2: Function to Detect Sampling Rate ---
def detect_sampling_rate(times: np.ndarray) -> int:
    """
    Detects the sampling rate in Hz from a NumPy array of time values (in seconds).

    Args:
        times (np.ndarray): NumPy array of time values in seconds.

    Returns:
        int: The detected sampling rate in Hz, rounded to the nearest integer.

    Raises:
        ValueError: If the times array is too short or if a consistent sampling
                    rate cannot be determined.
    """
    if len(times) < 2:
        raise ValueError("Time array must contain at least two points to detect sampling rate.")

    # Calculate differences between consecutive time points
    dt_values = np.diff(times)

    # Filter out near-zero differences if any (e.g., from identical timestamps)
    dt_values = dt_values[dt_values > 1e-9] # Small epsilon to avoid division by zero

    if len(dt_values) == 0:
        raise ValueError("No valid time differences found to calculate sampling rate.")

    # Find the most frequent time step using a histogram-like approach for robustness
    # Round to a sufficient number of decimal places (e.g., 7 for 600 Hz time steps)
    dt_counts = {}
    for dt_val in dt_values:
        rounded_dt = round(dt_val, 7) # Round to 7 decimal places for consistent binning
        dt_counts[rounded_dt] = dt_counts.get(rounded_dt, 0) + 1
    
    if not dt_counts:
        raise ValueError("No distinct time steps found.")

    # Get the most frequent dt
    most_frequent_dt = max(dt_counts, key=dt_counts.get)
    
    if most_frequent_dt == 0:
        raise ValueError("Detected time step is zero, cannot determine sampling rate.")

    # Sampling rate is 1 / most_frequent_dt, rounded to the nearest integer
    present_sampling_rate = int(round(1.0 / most_frequent_dt))

    return present_sampling_rate

# --- Step 3: Function to Downsample Data ---
def downsample_data(data: np.ndarray, present_sampling_rate: int, target_sampling_rate: int = 400) -> np.ndarray:
    """
    Downsamples a 1D NumPy array using scipy.signal.resample_poly.

    Args:
        data (np.ndarray): The 1D NumPy array to be downsampled.
        present_sampling_rate (int): The current sampling rate of the data in Hz.
        target_sampling_rate (int, optional): The desired sampling rate in Hz. Defaults to 400.

    Returns:
        np.ndarray: The downsampled NumPy array.
    
    Raises:
        ValueError: If target_sampling_rate is greater than present_sampling_rate,
                    or if resampling factors cannot be determined.
    """
    if target_sampling_rate > present_sampling_rate:
        raise ValueError("Target sampling rate must be less than or equal to the present sampling rate for downsampling.")
    
    # This check is now handled at a higher level in batch_process_data to control saving.
    # If this function is called, it means actual downsampling is expected.
    # if target_sampling_rate == present_sampling_rate:
    #     print(f"  Sampling rate is already {present_sampling_rate} Hz. No downsampling needed.")
    #     return data

    # Calculate resampling factors (up and down)
    # up / down = target_sampling_rate / present_sampling_rate
    # Use math.gcd to simplify the fraction
    common_divisor = math.gcd(target_sampling_rate, present_sampling_rate)
    up = target_sampling_rate // common_divisor
    down = present_sampling_rate // common_divisor

    if down == 0: # Should not happen if present_sampling_rate > 0
        raise ValueError("Invalid sampling rates provided; 'down' factor is zero.")

    print(f"  Downsampling from {present_sampling_rate} Hz to {target_sampling_rate} Hz (factors: up={up}, down={down})...")
    downsampled_data = signal.resample_poly(data, up, down)
    return downsampled_data

# --- Step 4: Check and Trim Downsampled Data Lengths ---
# This function is retained for robustness but should ideally not be needed
# if resample_poly is used correctly on initially same-length arrays.
def ensure_uniform_length(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """
    Ensures all NumPy arrays in a list have the same length by trimming
    them to the minimum length found among them.

    Args:
        arrays (list[np.ndarray]): A list of NumPy arrays.

    Returns:
        list[np.ndarray]: A new list of arrays, all trimmed to the minimum length.
                          Returns original list if all lengths are already uniform.
    """
    if not arrays:
        return []

    min_len = min(len(arr) for arr in arrays)
    
    # Check if trimming is actually needed
    if all(len(arr) == min_len for arr in arrays):
        return arrays # All arrays already have uniform length

    print(f"  Warning: Downsampled arrays have inconsistent lengths. Trimming to min length {min_len}.")
    trimmed_arrays = [arr[:min_len] for arr in arrays]
    return trimmed_arrays

# --- Step 5: Function to Generate Downsampled Times ---
def generate_downsampled_times(num_points: int, target_sampling_rate: int) -> np.ndarray:
    """
    Generates a NumPy array of time values for downsampled data, starting at 0.

    Args:
        num_points (int): The number of data points in the downsampled array.
        target_sampling_rate (int): The target sampling rate in Hz.

    Returns:
        np.ndarray: A NumPy array of time values (in seconds).
    """
    # Create time steps from 0 up to num_points * (1/target_sampling_rate)
    # Using np.float64 for precision
    downsampled_times = np.arange(num_points, dtype=np.float64) / target_sampling_rate
    return downsampled_times

# --- Step 6: Function to Generate Downsampled Output Filepath ---
def generate_downsampled_output_filepath(input_filepath: str, target_sampling_rate: int, output_dir: str = None) -> str:
    """
    Generates the output filepath for the downsampled .atf file, retaining
    descriptive filename elements and updating the sampling rate/keyword.

    Args:
        input_filepath (str): The original input .atf file path.
        target_sampling_rate (int): The target sampling rate in Hz.
        output_dir (str, optional): The directory where the new file will be saved.
                                    If None, uses the input file's directory.

    Returns:
        str: The full path to the new downsampled .atf file.
    """
    basename = os.path.basename(input_filepath)
    filename_without_ext, ext = os.path.splitext(basename)

    # Regex to capture parts for flexible filename construction
    # Group 1: Everything before the Hz part (e.g., '11n09001-guesthost_Tyr-70_mV')
    # Group 2: The Hz part itself (e.g., '-600_Hz')
    # Group 3: Everything after the Hz part (e.g., '-rpt_1' or empty string)
    match = re.match(r'(.*)(-\d+_Hz)(.*)', filename_without_ext)

    if match:
        prefix = match.group(1)
        suffix_after_hz = match.group(3) # This might include '-rpt_X'

        new_hz_part = f"-{target_sampling_rate}_Hz"
        downsampled_keyword = "-downsampled"

        # Combine parts: prefix + new_hz_part + downsampled_keyword + suffix_after_hz
        # Logic to avoid double hyphens if suffix_after_hz starts with '-'
        # and downsampled_keyword also ends with a hyphen.
        # This handles cases like -downsampled- -rpt_1 -> -downsampled-rpt_1
        if downsampled_keyword.endswith('-') and suffix_after_hz.startswith('-'):
            new_filename_without_ext = f"{prefix}{new_hz_part}{downsampled_keyword}{suffix_after_hz[1:]}"
        else:
            new_filename_without_ext = f"{prefix}{new_hz_part}{downsampled_keyword}{suffix_after_hz}"
        
    else:
        # If no -XXX_Hz pattern found, just append new Hz and downsampled at the end
        new_filename_without_ext = f"{filename_without_ext}-{target_sampling_rate}_Hz-downsampled"

    new_filename = new_filename_without_ext + ext

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, new_filename)
    else:
        return os.path.join(os.path.dirname(input_filepath), new_filename)


# --- Step 7: Function to Save Downsampled ATF ---
def save_downsampled_atf(output_filepath: str,
                         downsampled_times: np.ndarray,
                         downsampled_current: np.ndarray,
                         downsampled_voltage: np.ndarray,
                         header_lines: list[str]):
    """
    Saves the downsampled time, current, and voltage data to a new .atf file,
    preserving the original header and specified numerical precision for each column.

    Args:
        output_filepath (str): The full path to the output .atf file.
        downsampled_times (np.ndarray): NumPy array of downsampled time values.
        downsampled_current (np.ndarray): NumPy array of downsampled current values.
        downsampled_voltage (np.ndarray): NumPy array of downsampled voltage values.
        header_lines (list[str]): List of strings representing the original ATF header lines.

    Raises:
        IOError: If there's an issue writing the file.
        ValueError: If data arrays have inconsistent lengths.
    """
    if not (len(downsampled_times) == len(downsampled_current) == len(downsampled_voltage)):
        raise ValueError("Downsampled time, current, and voltage arrays must all have the same length for saving.")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            # Write header lines, preserving original formatting
            for line in header_lines:
                f.write(line + '\n')

            # Write data rows with specified precision for each column
            for i in range(len(downsampled_times)):
                # Time: 7 decimal places
                time_str = f"{downsampled_times[i]:.7f}"
                # Current: 5 decimal places
                current_str = f"{downsampled_current[i]:.5f}"
                # Voltage: 4 decimal places
                voltage_str = f"{downsampled_voltage[i]:.4f}"
                f.write(f"{time_str}\t{current_str}\t{voltage_str}\n")
        print(f"Downsampled ATF data successfully saved to: {output_filepath}")
    except Exception as e:
        raise IOError(f"Error saving downsampled ATF data to '{output_filepath}': {e}")

# --- Step 8: Optional Function to Plot Time Series Data ---
def plot_time_series_data(raw_time: np.ndarray, raw_current: np.ndarray,
                          downsampled_time: np.ndarray, downsampled_current: np.ndarray,
                          title: str = "Raw vs. Downsampled Current Data"):
    """
    Plots raw and downsampled current vs. time data for visual inspection.

    Args:
        raw_time (np.ndarray): Time array for raw data.
        raw_current (np.ndarray): Current array for raw data.
        downsampled_time (np.ndarray): Time array for downsampled data.
        downsampled_current (np.ndarray): Current array for downsampled data.
        title (str, optional): Title for the overall plot.
    """
    plt.figure(figsize=(12, 8))

    # Top plot: Raw data
    plt.subplot(2, 1, 1) # 2 rows, 1 column, 1st plot
    plt.plot(raw_time, raw_current, label='Raw Data', color='blue', alpha=0.7)
    plt.title(f"{title}\nRaw Data (Original Sampling Rate)")
    plt.ylabel("Current (pA)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # Bottom plot: Downsampled data
    plt.subplot(2, 1, 2) # 2 rows, 1 column, 2nd plot
    plt.plot(downsampled_time, downsampled_current, label='Downsampled Data', color='red', alpha=0.8)
    plt.title(f"Downsampled Data (Target Sampling Rate)")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (pA)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout() # Adjusts subplot params for a tight layout
    plt.show()

# --- Step 9: Batch Processor Function ---
def batch_process_data(input_filepaths_list: list[str],
                       target_sampling_rate: int = 400,
                       output_dir: str = None,
                       raw_header_idx: int = 9, # Default for raw snippet
                       view_plot: bool = False,
                       log_file_name: str = "downsampling_log.csv"):
    """
    Batch processes a list of .atf files, downsampling their data if necessary,
    and saving the results as new .atf files. Generates a log file of the process.

    Args:
        input_filepaths_list (list[str]): A list of full paths to the input .atf files.
        target_sampling_rate (int, optional): The desired sampling rate for output files in Hz.
                                              Defaults to 400.
        output_dir (str, optional): The directory where the output .atf files
                                    should be saved. If None, output files are saved
                                    in the same directory as their respective input files.
                                    Defaults to None.
        raw_header_idx (int, optional): The 0-indexed row number for the header in raw .atf files.
                                        Defaults to 9.
        view_plot (bool, optional): If True, plots raw vs. downsampled current data for
                                    each processed file where downsampling occurred.
                                    Defaults to False.
        log_file_name (str, optional): The name of the CSV log file. If it exists,
                                       an increasing numerical index will be appended.
                                       Defaults to "downsampling_log.csv".
    """
    print(f"\n--- Starting Batch Processing of {len(input_filepaths_list)} files ---")

    log_data = [] # To store data for the CSV log file

    # Determine log file path with unique name
    base_log_name, log_ext = os.path.splitext(log_file_name)
    if output_dir:
        log_dir = output_dir
    else:
        # Use the directory of the first input file if output_dir is None,
        # otherwise current working directory.
        if input_filepaths_list:
            log_dir = os.path.dirname(input_filepaths_list[0]) if os.path.dirname(input_filepaths_list[0]) else os.getcwd()
        else:
            log_dir = os.getcwd() # Fallback if list is empty
    
    log_filepath = os.path.join(log_dir, log_file_name)
    counter = 0
    while os.path.exists(log_filepath):
        counter += 1
        log_file_name = f"{base_log_name}_{counter}{log_ext}"
        log_filepath = os.path.join(log_dir, log_file_name)

    for input_filepath in input_filepaths_list:
        detected_hz = "N/A" # For log
        downsampled_hz = "N/A" # For log
        output_filepath_for_log = "N/A" # For log, will be updated or set to NOT_SAVED
        process_status = "Failed" # For log

        try:
            print(f"\nProcessing: {input_filepath}")

            # Step 1: Load ATF data
            times_raw, current_raw, voltage_raw, header_lines = load_atf(input_filepath, header_row_index=raw_header_idx)
            
            # Step 2: Detect sampling rate
            detected_hz = detect_sampling_rate(times_raw)
            print(f"  Detected sampling rate: {detected_hz} Hz")

            # Conditional logic for processing and saving
            if detected_hz > target_sampling_rate:
                # Actual downsampling needed
                downsampled_hz = target_sampling_rate

                # Step 3: Downsample data
                downsampled_current = downsample_data(current_raw, detected_hz, target_sampling_rate)
                downsampled_voltage = downsample_data(voltage_raw, detected_hz, target_sampling_rate)
                
                # Step 4: Ensure current and voltage are same length post-resampling
                [downsampled_current, downsampled_voltage] = \
                    ensure_uniform_length([downsampled_current, downsampled_voltage])
                
                # Step 5: Generate NEW downsampled times based on the length of processed data
                downsampled_times = generate_downsampled_times(len(downsampled_current), target_sampling_rate)

                # Step 6: Generate downsampled output filepath
                output_filepath_for_log = generate_downsampled_output_filepath(input_filepath, target_sampling_rate, output_dir)
                
                # Step 7: Save processed ATF
                save_downsampled_atf(output_filepath_for_log, downsampled_times, downsampled_current, downsampled_voltage, header_lines)
                
                process_status = "Success (Downsampled)"
                
                # Step 8: Plot (optional) - only for truly downsampled files
                if view_plot:
                    plot_time_series_data(times_raw, current_raw, downsampled_times, downsampled_current,
                                          title=f"Downsampling for {os.path.basename(input_filepath)}")

            elif detected_hz < target_sampling_rate:
                # Detected rate is lower than target, so we skip upsampling
                print(f"  Note: Detected sampling rate ({detected_hz} Hz) is lower than target ({target_sampling_rate} Hz). Skipping processing.")
                downsampled_hz = detected_hz # Log the actual detected rate
                process_status = "Skipped (Lower Hz)"
                output_filepath_for_log = "NOT_SAVED (Undersampled)"

            else: # detected_hz == target_sampling_rate
                # Detected rate matches target, so we skip re-saving
                print(f"  Note: Detected sampling rate ({detected_hz} Hz) matches target ({target_sampling_rate} Hz). Skipping processing.")
                downsampled_hz = detected_hz # Log the actual detected rate
                process_status = "Skipped (Same Hz)"
                output_filepath_for_log = "NOT_SAVED (Already Target Hz)"

        except FileNotFoundError as e:
            print(f"  Error: {e}")
            process_status = "File Not Found"
            output_filepath_for_log = "ERROR" # Indicate error in output path
        except KeyError as e:
            print(f"  Error: {e}")
            process_status = "Missing Column"
            output_filepath_for_log = "ERROR"
        except ValueError as e:
            print(f"  Error: {e}")
            process_status = "Data Error"
            output_filepath_for_log = "ERROR"
        except Exception as e:
            print(f"  An unexpected error occurred: {e}")
            process_status = "Unexpected Error"
            output_filepath_for_log = "ERROR"
        finally:
            log_data.append({
                'input_filepath': input_filepath,
                'detected_Hz': detected_hz,
                'downsampled_Hz': downsampled_hz,
                'output_filepath': output_filepath_for_log,
                'status': process_status
            })

    # Save log file
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(log_filepath, index=False)
    print(f"\nBatch processing log saved to: {log_filepath}")
    print("\n--- Batch Processing Complete ---")

if __name__ == "__main__":
    output_dir = "./PA_F427A/guesthost_TrpDL/"
    
    filepaths_text_block = """
./PA_F427A/guesthost_TrpDL/11o27002-guesthost_TrpDL-70_mV-F427A-600_Hz-rpt_1.atf
./PA_F427A/guesthost_TrpDL/11o27002-guesthost_TrpDL-70_mV-F427A-600_Hz-rpt_2.atf
./PA_F427A/guesthost_TrpDL/11o27002-guesthost_TrpDL-70_mV-F427A-600_Hz-rpt_3.atf
./PA_F427A/guesthost_TrpDL/11o27003-guesthost_TrpDL-70_mV-F427A-600_Hz-rpt_1.atf
./PA_F427A/guesthost_TrpDL/11o27003-guesthost_TrpDL-70_mV-F427A-600_Hz-rpt_2.atf
./PA_F427A/guesthost_TrpDL/11o27004-guesthost_TrpDL-70_mV-F427A-600_Hz-rpt_1.atf
./PA_F427A/guesthost_TrpDL/11o27004-guesthost_TrpDL-70_mV-F427A-600_Hz-rpt_2.atf
./PA_F427A/guesthost_TrpDL/11o28000-guesthost_TrpDL-70_mV-F427A-600_Hz.atf
"""

    input_filepaths_list = [line.strip() for line in filepaths_text_block.splitlines() if line.strip()]

    target_sampling_rate = 400
    
    batch_process_data(input_filepaths_list,
                       target_sampling_rate = target_sampling_rate,
                       output_dir = output_dir,
                       raw_header_idx = 9,
                       view_plot = False,
                       log_file_name = "downsampling_log.csv")
