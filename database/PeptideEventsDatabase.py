import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
from typing import Optional, Any, List, Union
import json
import csv
import time
import hashlib

# Imports from segmentation/processing functions (e.g., segmentation_core.py)
from .segmentation_core import (
    load_stream,
    correct_baseline_and_drift,
    apply_median_filter,
    apply_bessel_filter,
    segment_translocations,
    compute_event_level_features,
    compute_global_features,
    prepare_ml_dl_data
)

# Import PeptideData and PeptideDatabase
from .PeptideDatabase import PeptideData, PeptideDatabase

# --- PeptideTranslocationEvents Class Definitions ---
@dataclass
class ProcessedPeptideData:
    # Fields without default values (must be provided during initialization)
    raw_record_id: str # The _id from the PeptideData record that this was derived from
    peptide_name: str
    data_file: str # Original raw data filename (e.g., peptide_A_simulated_...)
    processed_file: str # Path to the processed (PKL) file - NOW BEFORE OPTIONALS

    # Fields with default values or Optional (can be omitted during initialization)
    _id: str = field(init=False) # Unique ID for this processed record
    date: str = field(init=False) # Date when this processed record was created
    data_path: Optional[str] = None # Path to the raw data file (from PeptideData)
    processed_path: Optional[str] = None # Directory where the processed PKL is saved
    processing_params: dict = field(default_factory=dict)
    user: Optional[str] = None
    simulation: Optional[bool] = None
    experimental: Optional[bool] = None
    nanopore_name: Optional[str] = None
    voltage: Optional[float] = None
    buffer: Optional[str] = None
    
    # --- NEW METADATA FIELDS ---
    ph_cis: Optional[float] = None
    ph_trans: Optional[float] = None
    salt: Optional[str] = None
    salt_conc: Optional[float] = None
    # ---------------------------
    
    peptide_conc: Optional[float] = None
    raw_data_file: Optional[str] = None
    time_sampling: Optional[float] = None
    added_noise: Optional[str] = None
    comments: str = ""

    def __post_init__(self):
        if not hasattr(self, '_id') or self._id is None:
            self._id = str(uuid.uuid4())
        if not hasattr(self, 'date') or self.date is None:
            self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def from_dict(cls, data: dict):
        init_args = {}
        for f in cls.__dataclass_fields__.values():
            if f.init:
                if f.name in data:
                    init_args[f.name] = data[f.name]
                elif f.default is not field.MISSING:
                    init_args[f.name] = f.default
                elif f.default_factory is not field.MISSING:
                    init_args[f.name] = f.default_factory()
                elif hasattr(f.type, '__origin__') and f.type.__origin__ is Optional:
                    init_args[f.name] = None
        instance = cls(**init_args)
        if '_id' in data:
            instance._id = data['_id']
        if 'date' in data:
            instance.date = data['date']
        return instance

    
class PeptideTranslocationEvents:
    """
    A class to encapsulate the processing, segmentation, feature extraction,
    and management of translocation events from raw peptide data streams.
    An instance of this class represents a *specific set* of processed events
    derived from a raw data record using a given set of parameters.
    """

    def __init__(self,
                 raw_data_record: PeptideData, # Pass the full PeptideData object
                 processing_params: dict):
        """
        Initializes the PeptideTranslocationEvents object with the raw data record
        and processing parameters.

        Args:
            raw_data_record (PeptideData): The PeptideData dataclass instance
                                            from the PeptideDatabase.
            processing_params (dict): A dictionary of parameters for processing,
                                      e.g., {'high_pass_cutoff_frequency': 0.5,
                                            'min_event_duration_ms': 5, ...}
        """
        self.raw_record = raw_data_record
        self.processing_params = processing_params

        self._event_data = None
        self._feature_names = None
        self._processed_file_path = None

        print(f"Initialized PeptideTranslocationEvents for raw record ID: {self.raw_record._id} "
              f"({self.raw_record.peptide_name}) with parameters: {self.processing_params}")

    def process_stream(self, output_dir: str = './processed_data') -> Optional[ProcessedPeptideData]:
        """
        Processes a raw peptide translocation stream to segment events,
        compute features, and prepare data for ML/DL.

        Args:
            output_dir (str): Directory where the processed PKL file will be saved.

        Returns:
            Optional[ProcessedPeptideData]: A ProcessedPeptideData instance if successful,
                                            otherwise None.
        """
        raw_csv_filepath = os.path.join(self.raw_record.data_path, self.raw_record.data_file)
        
        # Serialize processing_params to a sorted JSON string for consistent hashing
        params_string = json.dumps(self.processing_params, sort_keys=True)
        param_hash = hashlib.sha256(params_string.encode('utf-8')).hexdigest()[:16]

        sanitized_peptide_name = "".join(
            c if c.isalnum() else '_' for c in self.raw_record.peptide_name
        ).replace('__', '_').strip('_')

        processed_filename = f"{sanitized_peptide_name}_{self.raw_record._id[:8]}_{param_hash}.pkl"
        output_pickle_filepath = os.path.join(output_dir, processed_filename)
        
        self._processed_file_path = output_pickle_filepath

        print(f"\n--- Processing {self.raw_record.peptide_name} (ID: {self.raw_record._id}) "
              f"from {raw_csv_filepath} ---")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        try:
            raw_times, raw_current, raw_states = load_stream(raw_csv_filepath)
        except FileNotFoundError:
            print(f"Error: Input file not found for {self.raw_record.peptide_name} (ID: {self.raw_record._id}): {raw_csv_filepath}. Aborting processing.")
            return None
        except Exception as e:
            print(f"Error loading stream for {self.raw_record.peptide_name} (ID: {self.raw_record._id}) from {raw_csv_filepath}: {e}. Aborting processing.")
            return None

        if raw_states.size == 0:
            print(f"No data loaded or processed successfully for {self.raw_record.peptide_name} (ID: {self.raw_record._id}). Aborting processing.")
            return None

        try:
            # Apply high-pass filter and baseline correction first
            filtered_current = correct_baseline_and_drift(
                raw_times,
                raw_current,
                high_pass_cutoff_frequency=self.processing_params.get('high_pass_cutoff_frequency', 0.5),
                filter_order=self.processing_params.get('filter_order', 3),
                polynomial_degree=self.processing_params.get('polynomial_degree', 2),
                apply_polynomial_correction=self.processing_params.get('apply_polynomial_correction', True)
            )
        except Exception as e:
            print(f"Error during baseline correction for {self.raw_record.peptide_name}: {e}. Aborting processing.")
            return None

        # --- NEW LOW-PASS FILTERING LOGIC ---
        current_for_segmentation = filtered_current # Start with high-pass filtered data
        low_pass_type = self.processing_params.get('low_pass_filter_type', 'none')
        low_pass_params = self.processing_params.get('low_pass_filter_params', {})
        sampling_rate = self.processing_params.get('sampling_rate_hz', 1000)

        if low_pass_type == 'bessel':
            try:
                bessel_config = low_pass_params.get('bessel', {})
                cutoff = bessel_config.get('cutoff_hz')
                order = bessel_config.get('order', 4) # Default order 4

                if cutoff is None:
                    print("Warning: 'low_pass_filter_type' is 'bessel' but 'cutoff_hz' not found in params. Skipping Bessel filter.")
                else:
                    print(f"Applying Bessel low-pass filter (Cutoff: {cutoff} Hz, Order: {order})")
                    current_for_segmentation = apply_bessel_filter(current_for_segmentation, cutoff, sampling_rate, order)
            except Exception as e:
                print(f"Error applying Bessel filter: {e}. Skipping Bessel filter.")

        elif low_pass_type == 'median':
            try:
                median_config = low_pass_params.get('median', {})
                window_size = median_config.get('window_size')

                if window_size is None:
                    print("Warning: 'low_pass_filter_type' is 'median' but 'window_size' not found in params. Skipping median filter.")
                else:
                    print(f"Applying median filter with window size: {window_size}")
                    current_for_segmentation = apply_median_filter(current_for_segmentation, window_size)
            except ValueError as ve:
                print(f"Error applying median filter: {ve}. Skipping median filter.")
            except Exception as e:
                print(f"Unexpected error during median filtering: {e}. Skipping median filter.")

        elif low_pass_type == 'none':
            print("No additional low-pass filter applied.")
        else:
            print(f"Warning: Unknown 'low_pass_filter_type': {low_pass_type}. No additional low-pass filter applied.")
        # --- END NEW LOW-PASS FILTERING LOGIC ---

        # Scale the current (apply to `current_for_segmentation` after all filtering)
        if current_for_segmentation.size > 0:
            # Next logic deals with different cases regarding the sign distribution of the signal when different offsets are applied
            if np.max(current_for_segmentation) > 0: # If the max current is positive (or very close to 0)
                print("Case 2: Current trace straddles 0 pA. Shifting to all negative values.")
                max_current_value = np.max(current_for_segmentation)
                # Subtract this max value to shift the entire trace so that this point becomes 0 pA,
                # and all other values become negative (or more negative).
                current_for_segmentation = current_for_segmentation - max_current_value
            else:
                print("Case 1: Current trace is already all negative. Proceeding as is.")
                # No offset needed, as it's already in the desired negative range.
                pass
            # Scale the absolute value of the signal
            scaler = MinMaxScaler()
            scaled_filtered_current = scaler.fit_transform(np.abs(current_for_segmentation).reshape(-1, 1)).flatten()
        else:
            print(f"Warning: Corrected current data for {self.raw_record.peptide_name} is empty, cannot scale. Aborting processing.")
            return None

        print(f"Current data for {self.raw_record.peptide_name} scaled.")

        try:
            event_currents, state_sequences, open_state = segment_translocations(
                scaled_filtered_current,
                raw_states,
                sampling_rate_hz=sampling_rate, # Use sampling_rate from params
                min_duration_ms=self.processing_params.get('min_event_duration_ms', 5)
            )
        except Exception as e:
            print(f"Error during segmentation for {self.raw_record.peptide_name}: {e}. Aborting processing.")
            return None

        if not state_sequences:
            print(f"No valid events segmented for {self.raw_record.peptide_name}. Aborting processing.")
            return None

        try:
            translocation_events_data = compute_event_level_features(event_currents, state_sequences, open_state)
        except Exception as e:
            print(f"Error computing event-level features for {self.raw_record.peptide_name}: {e}. Aborting processing.")
            return None

        if not translocation_events_data:
            print(f"No events with features computed for {self.raw_record.peptide_name}. Aborting processing.")
            return None

        try:
            translocation_events_data_with_globals = compute_global_features(translocation_events_data, open_state)
        except Exception as e:
            print(f"Error computing global features for {self.raw_record.peptide_name}: {e}. Aborting processing.")
            return None

        try:
            self._event_data, self._feature_names = prepare_ml_dl_data(translocation_events_data_with_globals, open_state)
        except Exception as e:
            print(f"Error preparing ML/DL data for {self.raw_record.peptide_name}: {e}. Aborting processing.")
            return None

        if not self._event_data:
            print(f"No data prepared for ML/DL input for {self.raw_record.peptide_name}. Aborting processing.")
            return None

        print(f"Saving prepared ML/DL data and feature names for {self.raw_record.peptide_name} to {output_pickle_filepath}...")
        try:
            output_data = {
                'events_data': self._event_data,
                'feature_names': self._feature_names
            }
            with open(output_pickle_filepath, 'wb') as outfile:
                pickle.dump(output_data, outfile)
            print(f"Prepared ML/DL data for {self.raw_record.peptide_name} saved successfully.")

            universal_processed_path = os.path.dirname(output_pickle_filepath).replace(os.sep, '/')
            
            processed_record = ProcessedPeptideData(
                raw_record_id=self.raw_record._id,
                peptide_name=self.raw_record.peptide_name,
                data_file=self.raw_record.data_file,
                processed_file=os.path.basename(output_pickle_filepath),
                data_path=self.raw_record.data_path,
                processed_path=universal_processed_path,
                processing_params=self.processing_params,
                user=self.raw_record.user,
                simulation=self.raw_record.simulation,
                experimental=self.raw_record.experimental,
                nanopore_name=self.raw_record.nanopore_name,
                voltage=self.raw_record.voltage,
                buffer=self.raw_record.buffer,
                
                # --- TRANSFER NEW METADATA FIELDS ---
                ph_cis=getattr(self.raw_record, 'ph_cis', None),
                ph_trans=getattr(self.raw_record, 'ph_trans', None),
                salt=getattr(self.raw_record, 'salt', None),
                salt_conc=getattr(self.raw_record, 'salt_conc', None),
                # ------------------------------------
                
                peptide_conc=self.raw_record.peptide_conc,
                raw_data_file=self.raw_record.raw_data_file,
                time_sampling=self.raw_record.time_sampling,
                added_noise=self.raw_record.added_noise,
                comments=f"Processed from raw record {self.raw_record._id} with these parameters."
            )
            return processed_record
        except Exception as e:
            print(f"Error saving prepared ML/DL data for {self.raw_record.peptide_name} to pickle file {output_pickle_filepath}: {e}")
            return None

    def get_events_data(self) -> list:
        if self._event_data is None:
            print("Warning: Events data not yet processed or loaded. Call process_stream() or load_events() first.")
        return self._event_data

    def get_feature_names(self) -> dict:
        if self._feature_names is None:
            print("Warning: Feature names not yet processed or loaded. Call process_stream() or load_events() first.")
        return self._feature_names

    def load_events(self, processed_pickle_filepath: str):
        try:
            with open(processed_pickle_filepath, 'rb') as infile:
                loaded_data = pickle.load(infile)
                self._event_data = loaded_data.get('events_data')
                self._feature_names = loaded_data.get('feature_names')
                self._processed_file_path = processed_pickle_filepath
            print(f"Successfully loaded events from {processed_pickle_filepath}")
            return True
        except FileNotFoundError:
            print(f"Error: Pickle file not found at {processed_pickle_filepath}.")
            return False
        except Exception as e:
            print(f"Error loading events from {processed_pickle_filepath}: {e}")
            return False

    def get_processed_file_path(self) -> str:
        return self._processed_file_path

    def get_processing_parameters(self) -> dict:
        return self.processing_params


# --- New PeptideEventsDatabase Class Definition ---
class PeptideEventsDatabase:
    def __init__(self, db_file="peptide_events_data.json"):
        self.db_file = db_file
        self._initialize_db()

    def _initialize_db(self):
        if not os.path.exists(self.db_file):
            with open(self.db_file, 'w') as f:
                json.dump([], f)

    def _read_db(self):
        try:
            with open(self.db_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: '{self.db_file}' is empty or corrupted. Reinitializing.")
            self._write_db([])
            return []

    def _write_db(self, data):
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def add_processed_record(self, processed_data: ProcessedPeptideData):
        """
        Adds a new ProcessedPeptideData record to the database,
        preventing duplicates based on raw_record_id and processing_params.
        """
        records = self._read_db()

        # Check for existing record with the same raw_record_id AND processing_params
        # We also compare the processed_file to be absolutely sure, as the hash forms part of it.
        for record_dict in records:
            if (record_dict.get('raw_record_id') == processed_data.raw_record_id and
                record_dict.get('processed_file') == processed_data.processed_file): # Processed_file includes param_hash
                print(f"Skipping add: Identical processed record for '{processed_data.peptide_name}' "
                      f"(Raw ID: {processed_data.raw_record_id}) with these parameters already exists "
                      f"in DB (Existing ID: {record_dict.get('_id')}).")
                # Optionally, you could return the existing record's ID here if you need it.
                return record_dict.get('_id') # Return existing ID

        # If no duplicate found, add the new record
        records.append(asdict(processed_data))
        self._write_db(records)
        print(f"Processed record for '{processed_data.peptide_name}' (Raw ID: {processed_data.raw_record_id}) added with ID: {processed_data._id}")
        return processed_data._id


    def retrieve_processed_records(self, query: dict = None) -> List[ProcessedPeptideData]:
        records = self._read_db()
        
        if query is None:
            return [ProcessedPeptideData.from_dict(record) for record in records]
        
        filtered_records = []
        for record_dict in records:
            match = True

            for key, value in query.items():
                if key == 'processing_params' and isinstance(value, dict):
                    # --- START DEBUG PRINTS ---
                    # print(f"\nDEBUG: Checking processing_params for record ID: {record_dict.get('_id', 'N/A')[:8]}...")
                    # print(f"DEBUG: Query processing_params: {value}")
                    record_proc_params = record_dict.get('processing_params', {})
                    # print(f"DEBUG: Record processing_params: {record_proc_params}")

                    all_sub_matches = True
                    for k_sub, v_sub in value.items():
                        record_sub_val = record_proc_params.get(k_sub)
                        is_match = (record_sub_val == v_sub)
                        # print(f"DEBUG:   Sub-key '{k_sub}': Query '{v_sub}' (type: {type(v_sub).__name__}), Record '{record_sub_val}' (type: {type(record_sub_val).__name__}) -> Match: {is_match}")
                        if not is_match:
                            all_sub_matches = False
                            break
                    # print(f"DEBUG: Overall processing_params match for this record: {all_sub_matches}")
                    # --- END DEBUG PRINTS ---

                    if not all_sub_matches: # Changed from `not all(...)` to use our debug variable
                        match = False
                        break
                elif key not in record_dict:
                    match = False
                    break

                record_value = record_dict[key]
                if isinstance(value, bool) and isinstance(record_value, str):
                    if str(record_value).lower() != str(value).lower():
                        match = False
                        break
                elif record_value != value:
                    match = False
                    break
            if match:
                filtered_records.append(ProcessedPeptideData.from_dict(record_dict))
        return filtered_records

    def delete_processed_record(self, record_id: str, delete_pkl_file: bool = True) -> bool:
        """
        Deletes a ProcessedPeptideData record from the database and optionally
        removes the associated PKL file.

        Args:
            record_id (str): The unique ID (_id) of the processed record to delete.
            delete_pkl_file (bool): If True, also attempts to delete the corresponding
                                    PKL file from the filesystem. Defaults to True.

        Returns:
            bool: True if the record was found and deleted (and PKL file if requested),
                  False otherwise.
        """
        # Assuming self._read_db() and self._write_db() are methods that handle
        # reading from and writing to your database (e.g., a JSON file or other storage).
        # Also assuming ProcessedPeptideData is a class you have defined elsewhere.
        # Make sure to import 'os' if it's not already imported in your class/module:
        # import os

        records = self._read_db()
        initial_len = len(records)
        record_to_delete = None
        record_index = -1

        for i, record_dict in enumerate(records):
            if record_dict.get('_id') == record_id:
                # Assuming ProcessedPeptideData has a from_dict class method
                record_to_delete = ProcessedPeptideData.from_dict(record_dict)
                record_index = i
                break

        if record_to_delete is None:
            print(f"No processed record found with ID: {record_id}")
            return False

        # Remove from database list
        del records[record_index]
        self._write_db(records)
        print(f"Processed record with ID: {record_id} removed from database.")

        # Optionally delete the PKL file
        if delete_pkl_file and record_to_delete.processed_path and record_to_delete.processed_file:
            pkl_filepath = os.path.join(record_to_delete.processed_path, record_to_delete.processed_file)
            pkl_filepath = os.path.normpath(pkl_filepath)
            if os.path.exists(pkl_filepath):
                try:
                    os.remove(pkl_filepath)
                    print(f"Associated PKL file deleted: {pkl_filepath}")
                except OSError as e:
                    print(f"Error deleting PKL file {pkl_filepath}: {e}")
                    return False # Indicate partial failure
            else:
                print(f"Warning: PKL file not found at expected path {pkl_filepath} for record ID {record_id}. Record still deleted from DB.")
        elif delete_pkl_file:
            print(f"Warning: Cannot delete PKL file for record ID {record_id}. processed_path or processed_file missing from record metadata.")

        return True

    
    def get_processed_file_path(self, processed_record_id: str) -> Optional[str]:
        records = self.retrieve_processed_records(query={'_id': processed_record_id})
        if records:
            record = records[0]
            if record.processed_path and record.processed_file:
                return os.path.join(record.processed_path, record.processed_file)
        return None