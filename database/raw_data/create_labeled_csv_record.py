import pandas as pd
import numpy as np
import os
import shutil 
import datetime 

# --- Function 1: Read Event Annotation Data ---
def read_atf_event_data(filepath: str, header_row_index: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a .atf file containing event annotations, extracts 'Level',
    'Event Start Time (ms)', and 'Event End Time (ms)' columns, and
    returns them as NumPy arrays.

    Args:
        filepath (str): The path to the .atf event annotation file.
        header_row_index (int, optional): The 0-indexed row number where the column
                                          names are located. Defaults to 2 (3rd line).

    Returns:
        tuple: A tuple containing three numpy arrays:
               (levels, event_start, event_end)
        Raises:
            FileNotFoundError: If the specified filepath does not exist.
            KeyError: If expected column names are not found in the file.
            ValueError: If data cannot be parsed or if the file is empty/malformed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Event file not found at '{filepath}'")
    
    try:
        df = pd.read_csv(filepath, sep='\t', header=header_row_index) 

        # Strip any potential leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()

        required_columns = [
            'Level',
            'Event Start Time (ms)',
            'Event End Time (ms)'
        ]

        # Check if all required columns are present
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise KeyError(f"Missing required columns in event file '{filepath}': {missing_cols}. Available: {df.columns.tolist()}")

        # Extract the desired columns and convert to numpy arrays
        levels = df['Level'].to_numpy()
        event_start = df['Event Start Time (ms)'].to_numpy()
        event_end = df['Event End Time (ms)'].to_numpy()

        return levels, event_start, event_end

    except pd.errors.EmptyDataError:
        raise ValueError(f"The event file '{filepath}' is empty or has no data after headers.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing event file '{filepath}': {e}")
    except Exception as e:
        # Re-raise the original exception for specific handling in batch_process_data
        raise e 

# --- Function 2: Read Raw Data ---
def read_atf_raw_data(filepath: str, header_row_index: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """
    Opens a raw Axon Text File (.atf), skips the multi-line header,
    reads the time and current data, and returns them as NumPy arrays.

    Args:
        filepath (str): The full path to the raw .atf file.
        header_row_index (int, optional): The 0-indexed row number where the column
                                          names are located. Defaults to 9 (10th line).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                        (time_in_seconds, current_in_pA).
    Raises:
        FileNotFoundError: If the specified filepath does not exist.
        KeyError: If expected column names are not found in the file.
        ValueError: If data cannot be parsed or if the file is empty/malformed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at '{filepath}'")

    try:
        df = pd.read_csv(filepath, sep='\t', header=header_row_index)

        # Rename columns to remove potential leading/trailing spaces and special characters
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(' #', '') # Remove '# ' from 'Trace #1 (pA)'
        df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores
        df.columns = df.columns.str.replace('[()]', '', regex=True) # Remove parentheses

        # Define the expected column names after cleaning
        time_col_name = "Time_s"
        current_col_name = "Trace1_pA" 

        # Check if the required columns exist
        if time_col_name not in df.columns:
            raise KeyError(f"Column '{time_col_name}' not found in raw data file: {filepath}. Available: {df.columns.tolist()}")
        if current_col_name not in df.columns:
            raise KeyError(f"Column '{current_col_name}' not found in raw data file: {filepath}. Available: {df.columns.tolist()}")

        # Extract 'Time (s)' and 'Trace #1 (pA)' columns and convert to numpy arrays
        time = df[time_col_name].to_numpy()
        current = df[current_col_name].to_numpy()

        return time, current

    except pd.errors.EmptyDataError:
        raise ValueError(f"The raw data file '{filepath}' is empty or has no data after headers.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing raw data file '{filepath}': {e}")
    except Exception as e:
        # Re-raise the original exception for specific handling in batch_process_data
        raise e 

# --- Function 3: Label Data ---
def label_data(time: np.ndarray,
               current: np.ndarray,
               levels: np.ndarray,
               event_start: np.ndarray,
               event_end: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Labels the raw current data with corresponding conductance state levels
    based on event start and end times.
    Unassigned points at the start or end of the trace (edge points) will be labeled with 0.
    The '0' state represents the closed nanopore.
    If unlabeled points are found internally (not at the edges), a ValueError is raised,
    indicating potential file corruption.

    Args:
        time (np.ndarray): NumPy array of time values (in seconds) from raw data.
        current (np.ndarray): NumPy array of current values (in pA) from raw data.
        levels (np.ndarray): NumPy array of conductance state levels.
        event_start (np.ndarray): NumPy array of event start times (in milliseconds).
        event_end (np.ndarray): NumPy array of event end times (in milliseconds).

    Returns:
        tuple[np.ndarray, int, int]: A tuple containing:
            - np.ndarray: A NumPy array of the same length as 'time' and 'current',
                          containing the assigned state level labels as integers.
                          Unassigned edge points are labeled with 0.
            - int: The number of unassigned edge points.
            - int: The number of unassigned internal points (will be 0 if no error).
    Raises:
        ValueError: If input arrays have inconsistent lengths, or if unlabeled
                    internal points are detected.
    """
    if not (len(time) == len(current)):
        raise ValueError("Time and current arrays must have the same length.")
    if not (len(levels) == len(event_start) == len(event_end)):
        raise ValueError("Levels, event_start, and event_end arrays must have the same length.")

    # Initialize the 'states' array with NaN (float type).
    states = np.full_like(time, np.nan, dtype=float)

    # Convert event_start and event_end from milliseconds to seconds
    event_start_sec = event_start / 1000.0
    event_end_sec = event_end / 1000.0

    # Iterate through each event and apply the corresponding level
    for i in range(len(levels)):
        level = levels[i]
        start_time = event_start_sec[i]
        end_time = event_end_sec[i]

        # Find indices in the 'time' array that fall within the current event window
        # Use a small epsilon for end_time to ensure inclusive range for floating point comparisons
        indices_in_event = (time >= start_time) & (time <= end_time + 1e-9) 

        # Assign the current level to the 'states' array for the identified indices
        states[indices_in_event] = level

    # --- Handle unlabeled points ---
    nan_indices = np.where(np.isnan(states))[0]
    unassigned_total_count = len(nan_indices)

    unassigned_edge_count = 0
    unassigned_internal_count = 0

    if unassigned_total_count > 0:
        non_nan_indices = np.where(~np.isnan(states))[0]

        if len(non_nan_indices) == 0:
            # Entire trace is unlabeled. Treat as internal corruption.
            unassigned_internal_count = unassigned_total_count
            raise ValueError(f"Unlabeled internal points detected: Entire trace ({unassigned_total_count} points) is unlabeled. This indicates severe file corruption.")
        
        first_labeled_idx = non_nan_indices[0]
        last_labeled_idx = non_nan_indices[-1]

        # Count NaNs before the first labeled point
        unassigned_edge_count += np.sum(nan_indices < first_labeled_idx)
        # Count NaNs after the last labeled point
        unassigned_edge_count += np.sum(nan_indices > last_labeled_idx)
        
        unassigned_internal_count = unassigned_total_count - unassigned_edge_count

        if unassigned_internal_count > 0:
            # Critical error: Internal unlabeled points found.
            # Fill them with -1 to clearly mark them if the script were to continue
            # before raising, but we'll raise immediately here.
            # states[np.where(np.isnan(states))[0]] = -1 # Or another distinct value for debugging
            raise ValueError(f"Unlabeled internal points detected: {unassigned_internal_count} points between indices {first_labeled_idx} and {last_labeled_idx} are unlabeled. This indicates potential file corruption.")
        else:
            # Only edge points are unlabeled, which is benign.
            print(f"  Info: {unassigned_edge_count} unlabeled edge points detected (labeled as 0).")
            # Fill these edge NaNs with 0
            states[np.where(np.isnan(states))[0]] = 0 
    
    # Convert to integer type at the very end
    return states.astype(int), unassigned_edge_count, unassigned_internal_count

# --- Function 4: Save Labeled Data ---
def save_labeled_data(time: np.ndarray,
                      current: np.ndarray,
                      states: np.ndarray,
                      output_filepath: str):
    """
    Saves the time, current, and labeled state data to a CSV file.

    Args:
        time (np.ndarray): NumPy array of time values.
        current (np.ndarray): NumPy array of current values.
        states (np.ndarray): NumPy array of labeled state values (expected to be integers).
        output_filepath (str): The full path to the output CSV file.
    Raises:
        ValueError: If input arrays have inconsistent lengths.
        IOError: If there's an issue writing the file.
    """
    if not (len(time) == len(current) == len(states)):
        raise ValueError("Time, current, and states arrays must all have the same length.")

    # Create a pandas DataFrame
    data = {
        'Time': time,
        'Current': current,
        'State': states
    }
    df = pd.DataFrame(data)

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        # Save the DataFrame to a CSV file without the index
        df.to_csv(output_filepath, index=False)
        print(f"Labeled data successfully saved to: {output_filepath}")
    except Exception as e:
        raise IOError(f"Error saving labeled data to '{output_filepath}': {e}")

# --- Function 5: Batch Processor ---
def batch_process_data(raw_filepaths_list: list[str],
                       events_suffix: str = '_events',
                       output_dir: str = None,
                       raw_header_idx: int = 9,
                       event_header_idx: int = 2,
                       log_file_name: str = "labeling_log.csv"): # New parameter for log file name
    """
    Batch processes a list of raw data .atf files to label conductance states
    and save the results as CSV files.

    For each raw data file, it infers the corresponding event annotation file
    based on a naming convention (e.g., adding '_events' before the extension).
    Generates a log file of the process.

    Args:
        raw_filepaths_list (list[str]): A list of full paths to the raw .atf files.
        events_suffix (str, optional): The suffix appended to the raw file name
                                       to form the event annotation file name.
                                       Defaults to '_events'.
        output_dir (str, optional): The directory where the output CSV files
                                    should be saved. If None, CSVs are saved
                                    in the same directory as their respective
                                    raw .atf files. Defaults to None.
        raw_header_idx (int, optional): The 0-indexed row number for the header
                                        in raw .atf files. Defaults to 9.
        event_header_idx (int, optional): The 0-indexed row number for the header
                                          in event .atf files. Defaults to 2.
        log_file_name (str, optional): The name of the CSV log file. If it exists,
                                       an increasing numerical index will be appended.
                                       Defaults to "labeling_log.csv".
    """
    print(f"\n--- Starting Batch Processing of {len(raw_filepaths_list)} files ---")

    log_data = [] # To store data for the CSV log file

    # Determine log file path with unique name
    base_log_name, log_ext = os.path.splitext(log_file_name)
    if output_dir:
        # Ensure log directory exists if output_dir is specified
        os.makedirs(output_dir, exist_ok=True)
        log_dir = output_dir
    else:
        # Use the directory of the first input file if raw_filepaths_list is not empty,
        # otherwise current working directory.
        if raw_filepaths_list:
            log_dir = os.path.dirname(raw_filepaths_list[0]) if os.path.dirname(raw_filepaths_list[0]) else os.getcwd()
        else:
            log_dir = os.getcwd() # Fallback if list is empty

    log_filepath = os.path.join(log_dir, log_file_name)
    counter = 0
    while os.path.exists(log_filepath):
        counter += 1
        log_file_name = f"{base_log_name}_{counter}{log_ext}"
        log_filepath = os.path.join(log_dir, log_file_name)

    for raw_filepath in raw_filepaths_list:
        # Initialize variables for the log entry for the current file
        event_filepath_attempted = "N/A"
        output_csv_filepath_for_log = "N/A"
        process_status = "Failed" # Default status, will be updated
        error_details = "N/A" # Default to N/A, will be filled on error or warning

        try:
            print(f"\nProcessing: {raw_filepath}")

            # 1. Construct event file path
            raw_basename = os.path.basename(raw_filepath)
            raw_filename_without_ext = os.path.splitext(raw_basename)[0]
            raw_file_dir = os.path.dirname(raw_filepath)

            event_filename = raw_filename_without_ext + events_suffix + ".atf"
            event_filepath_attempted = os.path.join(raw_file_dir, event_filename) # Store for log

            # 2. Construct output CSV file path
            output_csv_filename = raw_filename_without_ext + ".csv"
            if output_dir:
                # Ensure output directory exists (handled by save_labeled_data too, but good to have here)
                os.makedirs(output_dir, exist_ok=True)
                output_csv_filepath_for_log = os.path.join(output_dir, output_csv_filename)
            else:
                output_csv_filepath_for_log = os.path.join(raw_file_dir, output_csv_filename)

            # 3. Read raw data
            print(f"  Reading raw data from: {raw_filepath}")
            time_data, current_data = read_atf_raw_data(raw_filepath, header_row_index=raw_header_idx)

            # 4. Read event data
            print(f"  Reading event data from: {event_filepath_attempted}")
            levels, event_start, event_end = read_atf_event_data(event_filepath_attempted, header_row_index=event_header_idx)

            # 5. Label the data
            print("  Labeling data with conductance states...")
            # label_data now returns labeled_states, unassigned_edge_count, unassigned_internal_count
            labeled_states, unassigned_edge_count, unassigned_internal_count = label_data(time_data, current_data, levels, event_start, event_end)

            # 6. Save the labeled data
            save_labeled_data(time_data, current_data, labeled_states, output_csv_filepath_for_log)

            # Determine final status based on counts
            if unassigned_internal_count > 0: 
                # This case should technically be caught by the ValueError below,
                # but this serves as a robust fallback/assertion check.
                process_status = "Data Corruption (Internal Unlabeled)"
                error_details = f"{unassigned_internal_count} internal points not covered by event annotations."
            elif unassigned_edge_count > 0:
                process_status = "Success (with unlabeled edge points)"
                error_details = f"{unassigned_edge_count} edge points not covered by event annotations (labeled as 0)."
            else:
                process_status = "Success"
                error_details = "N/A" # No warnings/errors

        except FileNotFoundError as e:
            process_status = "File Not Found"
            error_details = str(e)
            print(f"  Error: {error_details}")
        except KeyError as e:
            process_status = "Missing Column"
            error_details = str(e)
            print(f"  Error: {error_details}")
        except ValueError as e: # This will now catch the specific "Unlabeled internal points detected" error
            if "Unlabeled internal points detected" in str(e):
                process_status = "Data Corruption (Internal Unlabeled)"
                error_details = str(e)
            else: # Other ValueErrors (e.g., empty file, parsing error)
                process_status = "Data Error"
                error_details = str(e)
            print(f"  Error: {error_details}")
        except IOError as e:
            process_status = "Save Error"
            error_details = str(e)
            print(f"  Error: {error_details}")
        except Exception as e:
            process_status = "Unexpected Error"
            error_details = str(e)
            print(f"  An unexpected error occurred: {error_details}")
        finally:
            # Append entry to log_data regardless of success or failure
            log_data.append({
                'input_raw_filepath': raw_filepath,
                'input_event_filepath': event_filepath_attempted,
                'output_csv_filepath': output_csv_filepath_for_log,
                'status': process_status,
                'error_details': error_details
            })

    # Save log file after all files are processed
    if log_data: # Only save if there's data to log
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(log_filepath, index=False)
        print(f"\nBatch processing log saved to: {log_filepath}")
    else:
        print("\nNo files were processed, no log file generated.")

    print("\n--- Batch Processing Complete ---")


# --- Example Usage ---
if __name__ == "__main__":
    output_dir = "/Users/bakrantz/Documents/python/database/raw_data/PA_F427Y/guesthost_Tyr/"

    raw_filepaths_text_block = """
/Users/bakrantz/Documents/python/database/raw_data/PA_F427Y/guesthost_Tyr/11d05001-guesthost_Tyr-70_mV-F427Y-600_Hz-rpt_1.atf
"""

    raw_filepaths_list = [line.strip() for line in raw_filepaths_text_block.splitlines() if line.strip()]

    # Run the batch processor
    batch_process_data(
        raw_filepaths_list=raw_filepaths_list,
        events_suffix='_events',
        output_dir=output_dir,
        raw_header_idx=9,
        event_header_idx=2,
        log_file_name="labeling_log.csv"
    )

