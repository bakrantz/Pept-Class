import os
from dataclasses import asdict # Needed for converting dataclass to dict for JSON saving
import numpy as np
import pickle

# Import the classes from your script
from PeptideEventsDatabase import ProcessedPeptideData, PeptideTranslocationEvents, PeptideEventsDatabase
from PeptideDatabase import PeptideData, PeptideDatabase # Assuming PeptideDatabase is in PeptideDatabase.py

# --- Configuration ---
RAW_DATA_DIR = './raw_data' # Directory where your raw CSV files are
PROCESSED_DATA_DIR = './processed_data' # Directory to save processed PKL files
PEPTIDE_DB_FILE = 'peptide_data.json' # Your raw data database file
EVENTS_DB_FILE = 'peptide_events_data.json' # Your processed events database file

# Make sure the raw data directory exists for the PeptideDatabase to find files
if not os.path.exists(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)
    print(f"Created raw data directory: {RAW_DATA_DIR}")

# --- Helper Function to Create Dummy CSV Data if not present ---
def create_dummy_csv(filepath, num_points=1000, sampling_rate_hz=1000):
    if not os.path.exists(filepath):
        print(f"Creating dummy CSV: {filepath}")
        times = np.arange(num_points) / sampling_rate_hz
        # Simulate some current data with a 'translocation'
        current = 100 + 10 * np.sin(2 * np.pi * 5 * times) # Base sine wave
        # Add a simulated translocation event (e.g., a dip in current)
        current[200:300] -= 50
        current[600:700] -= 30
        states = np.zeros(num_points, dtype=int)
        states[200:300] = 1 # Event state
        states[600:700] = 1 # Event state

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time (s)', 'Current (pA)', 'State'])
            for i in range(num_points):
                writer.writerow([times[i], current[i], states[i]])
    else:
        print(f"Dummy CSV already exists: {filepath}")


# --- Main Test Execution ---
if __name__ == "__main__":
    print(f"--- Starting Peptide Events Processing Test ---")

    # 1. Initialize PeptideDatabase and ensure it has raw data
    raw_db = PeptideDatabase(db_file=PEPTIDE_DB_FILE)
    all_raw_records = raw_db.retrieve_records()

    if not all_raw_records:
        print("\nNo raw data records found in PeptideDatabase. Populating with dummy data.")
        # Create dummy CSV and add a record to the raw database
        dummy_csv_filename = "peptide_A_simulated_dummy.csv"
        dummy_csv_filepath_full = os.path.join(RAW_DATA_DIR, dummy_csv_filename)
        create_dummy_csv(dummy_csv_filepath_full)

        sample_raw_data = PeptideData(
            peptide_name="Peptide A",
            data_file=dummy_csv_filename,
            data_path=RAW_DATA_DIR,
            user="test_user",
            simulation=True,
            experimental=False,
            nanopore_name="SimPore-1",
            voltage=100.0,
            buffer="PBS",
            time_sampling=0.001, # Corresponds to 1000 Hz
            comments="Simulated data for testing processing."
        )
        raw_db.add_peptide_data(sample_raw_data)
        all_raw_records = raw_db.retrieve_peptide_data() # Reload records

    if not all_raw_records:
        print("Failed to set up raw data. Exiting.")
        exit()

    print(f"\nFound {len(all_raw_records)} raw records in '{PEPTIDE_DB_FILE}'.")

    # To find all raw data records with Gaussian 0.1 std dev noise:
    noisy_records = raw_db.retrieve_records(query={'added_noise': 'gaussian 0.1 std dev'})

    # To find all noise-free idealized data (assuming 'None' means no synthetic noise):
    idealized_records = raw_db.retrieve_records(query={'added_noise': None})

    
    # Let's pick the first raw record for processing
    raw_record_to_process = noisy_records[0]
    print(f"Selected raw record for processing: '{raw_record_to_process.peptide_name}' (ID: {raw_record_to_process._id})")

    # 2. Define Processing Parameters
    processing_parameters = {
        'high_pass_cutoff_frequency': 0.5, # Adjusted for dummy data characteristics
        'filter_order': 3,
        'polynomial_degree': 2,
        'apply_polynomial_correction': True,
        'sampling_rate_hz': 1000,
        'min_event_duration_ms': 5, # Minimum event duration in milliseconds
        'apply_median_filter': True,
        'median_filter_window_size': 3 # Must be an odd number
    }
    print(f"\nProcessing parameters: {processing_parameters}")

    # 3. Initialize PeptideTranslocationEvents
    processor = PeptideTranslocationEvents(
        raw_data_record=raw_record_to_process,
        processing_params=processing_parameters
    )

    # 4. Process the Stream
    print("\n--- Initiating stream processing ---")
    processed_peptide_data_record = processor.process_stream(output_dir=PROCESSED_DATA_DIR)

    if processed_peptide_data_record:
        print("\n--- Stream processing completed successfully ---")
        print(f"Generated ProcessedPeptideData record ID: {processed_peptide_data_record._id}")
        print(f"Processed PKL file saved to: {processed_peptide_data_record.processed_path}/{processed_peptide_data_record.processed_file}")

        # 5. Add the processed record to PeptideEventsDatabase
        events_db = PeptideEventsDatabase(db_file=EVENTS_DB_FILE)
        new_processed_id = events_db.add_processed_record(processed_peptide_data_record)
        print(f"Processed record added to '{EVENTS_DB_FILE}' with ID: {new_processed_id}")

        # 6. Retrieve and Verify from PeptideEventsDatabase
        print("\n--- Verifying processed record retrieval from PeptideEventsDatabase ---")
        retrieved_records = events_db.retrieve_processed_records(query={'_id': new_processed_id})

        if retrieved_records:
            retrieved_processed_record = retrieved_records[0]
            print(f"Successfully retrieved processed record by ID: {retrieved_processed_record._id}")
            print(f"  Raw Record ID: {retrieved_processed_record.raw_record_id}")
            print(f"  Peptide Name: {retrieved_processed_record.peptide_name}")
            print(f"  Processing Params: {retrieved_processed_record.processing_params}")

            # Optionally, load the PKL file to inspect its content
            retrieved_pkl_path = events_db.get_processed_file_path(new_processed_id)
            if retrieved_pkl_path and os.path.exists(retrieved_pkl_path):
                print(f"Attempting to load PKL file from: {retrieved_pkl_path}")
                try:
                    with open(retrieved_pkl_path, 'rb') as f:
                        loaded_data = pickle.load(f)
                    print(f"PKL file loaded successfully. Contains {len(loaded_data.get('events_data', []))} events.")
                    print(f"Feature names available: {list(loaded_data.get('feature_names', {}).keys())}")
                except Exception as e:
                    print(f"Error loading processed PKL file: {e}")
            else:
                print(f"Processed PKL file not found at expected path: {retrieved_pkl_path}")
        else:
            print(f"Failed to retrieve processed record with ID: {new_processed_id}")
    else:
        print("\n--- Stream processing failed. No ProcessedPeptideData record was created. ---")

    print("\n--- Test Finished ---")
