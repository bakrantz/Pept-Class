import json
import os
import csv
from dataclasses import dataclass, asdict, field, fields, MISSING
from datetime import datetime
import uuid
from typing import Optional, Any, List, Union
import time

# --- PeptideData Class Definition ---
@dataclass
class PeptideData:
    """
    A dataclass to store peptide translocation experiment data.
    Fields with default values are optional during creation if not provided.
    """
    # Internal fields, not part of __init__
    _id: str = field(init=False)  # Unique serialized ID
    date: str = field(init=False) # Date when the record was created

    # Core required fields (adjust as truly essential for ALL record types)
    peptide_name: str
    data_file: str # The conductance state labeled .csv (time, current, state) file
    user: str 
    simulation: bool # True if data is from simulation
    experimental: bool # True if data is from experimental results

    # Optional fields with default None or empty string
    raw_data_file: Optional[str] = None # Usually a .atf text file
    data_path: Optional[str] = None # Path to the data_file
    peptide_sequence: Optional[str] = None # Note uppercase 1-letter codes denotes L-amino acids and lowercase denotes D-amino acids 
    nanopore_name: Optional[str] = None # Use 'PA' for wild type PA; for mutants of PA an example would be 'PA F427A'  
    voltage: Optional[float] = None # units of mV
    buffer: Optional[str] = None # Use 'UBB' universal bilayer buffer or 'SCB' single-channel buffer or other abbreviation if needed 
    peptide_conc: Optional[float] = None # units of nM
    time_sampling: Optional[float] = None # units of Hz
    added_noise: Optional[str] = None # Describes synthetic noise added to simulated data 
    comments: str = "" # Optional field, defaults to empty string

    def __post_init__(self):
        """
        Initializes _id and date fields after the dataclass is created.
        These are only set if they haven't been loaded from existing data (e.g., from DB).
        """
        if not hasattr(self, '_id') or self._id is None: # Ensure _id is generated if not present or None
            self._id = str(uuid.uuid4())
        if not hasattr(self, 'date') or self.date is None: # Ensure date is generated if not present or None
            self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a PeptideData instance from a dictionary, handling fields
        that are init=False by setting them after direct initialization.
        """
        init_args = {}
        for f in cls.__dataclass_fields__.values():
            if f.init: # Only include fields that are part of the __init__ method
                if f.name in data:
                    init_args[f.name] = data[f.name]
                elif f.default is not MISSING: # Use default if not present
                    init_args[f.name] = f.default
                elif f.default_factory is not MISSING: # Use default_factory if not present
                    init_args[f.name] = f.default_factory()
                elif hasattr(f.type, '__origin__') and f.type.__origin__ is Optional: # Handle Optional fields
                    init_args[f.name] = None # Explicitly set to None if missing and Optional

        instance = cls(**init_args)

        # Manually set fields that are init=False
        if '_id' in data:
            instance._id = data['_id']
        if 'date' in data:
            instance.date = data['date']
        
        return instance


# --- Database Manager Class ---
class PeptideDatabase:
    """
    Manages the local peptide data database (JSON file).
    """
    def __init__(self, db_file="peptide_data.json"): # This parameter now accepts the full path
        self.db_file = db_file
        self._initialize_db()

    def _initialize_db(self):
        """
        Creates the database file if it doesn't exist,
        and ensures its parent directory exists.
        """
        db_dir = os.path.dirname(self.db_file)
        
        if db_dir and not os.path.exists(db_dir):
            print(f"Creating directory for database: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)

        if not os.path.exists(self.db_file):
            print(f"Database file not found at {self.db_file}, creating empty JSON array.")
            with open(self.db_file, 'w') as f:
                json.dump([], f)

    def _read_db(self):
        """
        Reads all records from the database file.
        """
        try:
            with open(self.db_file, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"WARNING: '{self.db_file}' content is not a list ({type(data)}). Reinitializing.")
                    self._write_db([]) # Reinitialize with an empty list
                    return []
                return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"ERROR: '{self.db_file}' is empty, corrupted, or not found during _read_db: {e}. Reinitializing.")
            self._write_db([]) # Reinitialize with an empty list
            return []
        except Exception as e:
            print(f"ERROR: An unexpected error occurred in _read_db for {self.db_file}: {e}. Reinitializing.")
            self._write_db([]) # Reinitialize with an empty list
            return []


    def _write_db(self, data):
        """
        Writes data to the database file.
        """
        try:
            with open(self.db_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"ERROR: Failed to write to database file {self.db_file}: {e}")

    def add_record(self, peptide_data: 'PeptideData'):
        """
        Adds a new PeptideData record to the database.
        """
        records = self._read_db()
        records.append(asdict(peptide_data)) # Convert dataclass to dictionary
        self._write_db(records)
        print(f"Record for '{peptide_data.peptide_name}' added with ID: {peptide_data._id}")
        return peptide_data._id

    def retrieve_records(self, query: dict = None):
        """
        Retrieves records based on a query.
        Supports exact matches for fields, and range queries for numeric fields using MongoDB-like operators
        ($gt, $gte, $lt, $lte, $eq, $ne, $in).
        For example:
        query = {
            'peptide_name': 'guesthost_Ala',
            'voltage': {'$gt': 60, '$lte': 80},
            'experimental': True
        }
        Returns a list of PeptideData objects that match the query. If query is None, returns all records.
        """
        records = self._read_db()
        
        if query is None:
            return [PeptideData.from_dict(record) for record in records]
        
        filtered_records = []
        for i, record_dict in enumerate(records):
            match = True
            for key, query_value in query.items():
                if key not in record_dict:
                    match = False
                    break
                
                record_value = record_dict[key]

                if isinstance(query_value, dict):
                    # Handle comparison operators for range queries
                    for op, op_value in query_value.items():
                        if record_value is None: # Cannot compare None with numeric range
                            match = False
                            break
                        
                        # Attempt to convert record_value to the type of op_value for consistent comparison
                        try:
                            if isinstance(op_value, (int, float)) and not isinstance(record_value, (int, float)):
                                record_value = float(record_value) # Convert to float for numeric comparison
                            elif isinstance(op_value, bool) and not isinstance(record_value, bool):
                                record_value = str(record_value).lower() in ('true', '1', 't', 'y', 'yes')
                        except (ValueError, TypeError):
                            # If conversion fails, they are not comparable in the desired way
                            match = False
                            break

                        if op == "$eq":
                            if record_value != op_value:
                                match = False
                                break
                        elif op == "$ne":
                            if record_value == op_value:
                                match = False
                                break
                        elif op == "$gt":
                            if not (record_value is not None and record_value > op_value):
                                match = False
                                break
                        elif op == "$gte":
                            if not (record_value is not None and record_value >= op_value):
                                match = False
                                break
                        elif op == "$lt":
                            if not (record_value is not None and record_value < op_value):
                                match = False
                                break
                        elif op == "$lte":
                            if not (record_value is not None and record_value <= op_value):
                                match = False
                                break
                        elif op == "$in":
                            if record_value not in op_value:
                                match = False
                                break
                        else:
                            # Unknown operator
                            print(f"Warning: Unknown query operator '{op}' for key '{key}'. Skipping this query part.")
                            match = False # Treat as non-match for safety
                            break
                else:
                    # Handle exact match (non-dict query_value)
                    converted_record_value = record_value
                    # Add this block for numeric string conversion
                    if isinstance(query_value, (int, float)) and isinstance(record_value, str):
                        try:
                            converted_record_value = float(record_value) # Try converting string to float
                        except ValueError:
                            # If it can't be converted, they don't match
                            match = False
                            break
                    elif isinstance(query_value, bool) and isinstance(record_value, str):
                        converted_record_value = str(record_value).lower() in ('true', '1', 't', 'y', 'yes')
                    
                    if converted_record_value != query_value:
                        match = False
                        break
            if match:
                filtered_records.append(PeptideData.from_dict(record_dict))

        # This print statement is useful for debugging and will now show the correct count
        print(f"Found {len(filtered_records)} matching records for query: {query}")
        return filtered_records


    def edit_record(self, record_id: str, updates: dict):
        """
        Edits an existing record based on its _id.
        Updates format: {"field_name": "new_value"}
        """
        records = self._read_db()
        record_found = False
        for i, record in enumerate(records):
            if record.get("_id") == record_id:
                for key, value in updates.items():
                    if key in PeptideData.__dataclass_fields__:
                        field_type = PeptideData.__dataclass_fields__[key].type
                        try:
                            if hasattr(field_type, '__origin__') and field_type.__origin__ is Union and type(None) in field_type.__args__:
                                actual_type = [arg for arg in field_type.__args__ if arg is not type(None)][0]
                            else:
                                actual_type = field_type

                            if actual_type == float:
                                record[key] = float(value)
                            elif actual_type == bool:
                                record[key] = str(value).lower() in ('true', '1', 't', 'y', 'yes')
                            else: # Covers str, int, etc.
                                record[key] = value
                        except ValueError:
                            print(f"Warning: Could not convert update value '{value}' for field '{key}' to expected type {actual_type}. Skipping update for this field.")
                            continue
                    else:
                        print(f"Warning: Field '{key}' not found in record ID {record_id} or PeptideData definition.")
                        continue
                
                records[i] = record
                record_found = True
                break
        
        if record_found:
            self._write_db(records)
            print(f"Record ID {record_id} updated successfully.")
            return True
        else:
            print(f"Record with ID {record_id} not found.")
            return False

    def delete_records(self, record_ids: Union[str, List[str]]):
        """
        Deletes one or more records from the database based on their _id(s).
        Takes either a single _id string or a list of _id strings.
        """
        if isinstance(record_ids, str):
            record_ids = [record_ids]

        records = self._read_db()
        original_count = len(records)
        
        updated_records = [record for record in records if record.get("_id") not in record_ids]
        
        deleted_count = original_count - len(updated_records)
        
        if deleted_count > 0:
            self._write_db(updated_records)
            print(f"Successfully deleted {deleted_count} record(s).")
        else:
            print("No records found with the provided ID(s) to delete.")
        return deleted_count

    def import_from_csv(self, csv_file_path: str):
        """
        Imports multiple records from a CSV file into the database.
        The CSV header must match the PeptideData field names.
        Handles missing optional fields by setting them to None.
        """
        added_count = 0
        skipped_count = 0
        records = self._read_db()

        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            peptide_data_field_types = {f.name: f.type for f in fields(PeptideData) 
                                          if f.name not in ['_id', 'date']}

            for i, row in enumerate(reader):
                processed_row = {}
                type_conversion_errors = {}
                
                for field_name, field_type in peptide_data_field_types.items():
                    value_from_csv = row.get(field_name, "").strip()

                    actual_type = field_type.__args__[0] if hasattr(field_type, '__origin__') and field_type.__origin__ is Union and type(None) in field_type.__args__ else field_type

                    if value_from_csv == "":
                        if field_name == 'comments':
                            processed_row[field_name] = ""
                            continue
                        elif hasattr(field_type, '__origin__') and field_type.__origin__ is Union and type(None) in field_type.__args__ and actual_type != str:
                            processed_row[field_name] = None
                            continue

                    try:
                        if actual_type == float:
                            processed_row[field_name] = float(value_from_csv)
                        elif actual_type == bool:
                            processed_row[field_name] = str(value_from_csv).lower() in ('true', '1', 't', 'y', 'yes')
                        else: # Covers str, int, etc.
                            processed_row[field_name] = value_from_csv
                    except ValueError:
                        type_conversion_errors[field_name] = value_from_csv

                if type_conversion_errors:
                    error_details = ", ".join([f"{field}: '{val}'" for field, val in type_conversion_errors.items()])
                    print(f"Skipping row {i+2} (CSV line {i+2}) due to type conversion errors for fields: {error_details}")
                    skipped_count += 1
                    continue

                try:
                    new_peptide_data = PeptideData.from_dict(processed_row)
                    records.append(asdict(new_peptide_data))
                    added_count += 1
                except Exception as e:
                    print(f"Skipping row {i+2} (CSV line {i+2}) due to an unexpected error during record creation: {e} (Row data: {processed_row})")
                    skipped_count += 1
                
        self._write_db(records)
        print(f"\nCSV import complete. Added {added_count} records, skipped {skipped_count} records.")

    def delete_duplicate_records(self):
        """
        Identifies and deletes duplicate records, retaining the oldest (based on 'date' field).
        Two records are considered duplicates if all their fields (excluding '_id', 'date', and 'comments')
        are identical.
        """
        records = self._read_db()
        unique_records = {} # Key: tuple of identifying fields, Value: oldest record (dict)
        ids_to_delete = []

        ignored_fields = {'_id', 'date', 'comments'} 
        
        all_peptide_fields = {f.name for f in fields(PeptideData)}
        fields_to_compare = sorted(list(all_peptide_fields - ignored_fields))

        print("\n--- Checking for Duplicate Records ---")
        print(f"Fields used for comparison: {fields_to_compare}")

        for record_dict in records:
            comparison_key_values = []
            for field_name in fields_to_compare:
                comparison_key_values.append(record_dict.get(field_name))
            
            comparison_key = tuple(comparison_key_values)
            
            current_date_str = record_dict.get('date')
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S") if current_date_str else datetime.min

            if comparison_key in unique_records:
                existing_record = unique_records[comparison_key]
                existing_date = datetime.strptime(existing_record['date'], "%Y-%m-%d %H:%M:%S")

                if current_date < existing_date:
                    print(f"  Found older duplicate (retaining): ID {record_dict.get('_id')} (Date: {current_date_str}) "
                          f"vs existing ID {existing_record.get('_id')} (Date: {existing_record['date']}) - DELETING {existing_record.get('_id')}")
                    ids_to_delete.append(existing_record['_id'])
                    unique_records[comparison_key] = record_dict
                else: # This covers current_date >= existing_date
                    print(f"  Found newer or same-age duplicate (deleting): ID {record_dict.get('_id')} (Date: {current_date_str}) "
                          f"vs existing ID {existing_record.get('_id')} (Date: {existing_record['date']}) - DELETING {record_dict.get('_id')}")
                    ids_to_delete.append(record_dict['_id'])
            else:
                unique_records[comparison_key] = record_dict
            
        if ids_to_delete:
            print(f"\nIdentified {len(ids_to_delete)} duplicate(s) for deletion. IDs: {ids_to_delete}")
            self.delete_records(ids_to_delete)
        else:
            print("No duplicate records found.")

    def update_data_paths(self, base_search_directory: str):
        """
        Searches for data files within a base directory and its subdirectories
        and updates the 'data_path' field for matching records.
        Prioritizes the first found path if duplicates exist.
        """
        records = self._read_db()
        updated_count = 0
        
        print(f"\n--- Searching for data files in '{base_search_directory}' and subdirectories ---")

        found_files_map = {} # Key: filename (str), Value: full_directory_path (str)

        for root, dirs, files in os.walk(base_search_directory):
            for file in files:
                if file not in found_files_map:
                    found_files_map[file] = root

        for i, record_dict in enumerate(records):
            current_data_file = record_dict.get('data_file')
            
            if current_data_file and current_data_file in found_files_map:
                found_dir = found_files_map[current_data_file]
                if record_dict.get('data_path') != found_dir:
                    record_dict['data_path'] = found_dir
                    records[i] = record_dict
                    updated_count += 1
                    print(f"  Updated data_path for record ID {record_dict.get('_id')[:8]} ({record_dict.get('peptide_name')}/{current_data_file}) to: '{found_dir}'")
            elif current_data_file and record_dict.get('data_path') is None:
                    print(f"  Warning: Data file '{current_data_file}' for record ID {record_dict.get('_id')[:8]} not found in search path. 'data_path' remains None.")

        if updated_count > 0:
            self._write_db(records)
            print(f"--- Finished updating data paths. {updated_count} record(s) had their data_path updated. ---")
        else:
            print("--- No data_path updates needed or no files found. ---")
            
    def export_csv(self, csv_filepath: str):
        """
        Exports all database records as a CSV file.
        The header is created dynamically from PeptideData field names.
        """
        all_peptide_records = self.retrieve_records()
        if not all_peptide_records:
            print("No data to create CSV file.")
            return

        fieldnames = [f.name for f in fields(PeptideData)]

        try:
            with open(csv_filepath, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                writer.writeheader()

                for p_data in all_peptide_records:
                    writer.writerow(asdict(p_data))
            print(f"Successfully exported {len(all_peptide_records)} records to '{csv_filepath}'")
        except IOError as e:
            print(f"Error writing CSV file '{csv_filepath}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred during CSV export: {e}")

# --- Main Script to Calculate Recording Times ---

# Initialize the database. If your peptide_data.json is not in the same directory
# as this script, you'll need to provide the full path here.
# For example: db = PeptideDatabase("/path/to/your/peptide_data.json")
db = PeptideDatabase()

peptide_names = ["guesthost_Ala", "guesthost_Leu", "guesthost_Phe", "guesthost_Thr", "guesthost_Trp", "guesthost_TrpDL", "guesthost_Tyr"]

print("\n--- Find peptides in Database at specified range of peptide concentrations ---")
conc_low, conc_high = (5, 20)
print(f"Concentration range from {conc_low} nM to {conc_high} nM.")

# Dictionary to store total recording time per peptide
peptide_recording_times = {name: 0.0 for name in peptide_names}
total_recording_time_dataset = 0.0

for peptide_name in peptide_names:
    # Construct the query dictionary
    peptide_query = {
        'experimental': True,
        'nanopore_name': 'PA',
        'voltage': 70,
        'peptide_name': peptide_name,
        'time_sampling': 400,
        'peptide_conc': {'$gte': conc_low, '$lte': conc_high}
    }
    
    # Call retrieve_records with the enhanced query
    result_peptide_records = db.retrieve_records(peptide_query)

    print(f"\n--- Processing records for {peptide_name} ---")
    current_peptide_total_time = 0.0

    if not result_peptide_records:
        print(f"No records found for {peptide_name} matching the criteria.")
        continue

    for record in result_peptide_records:
        data_file = record.data_file
        data_path = record.data_path
        time_sampling = record.time_sampling # In Hz (samples per second)

        if not data_file or not data_path:
            print(f"  Skipping record {record._id[:8]} for {record.peptide_name}: 'data_file' or 'data_path' is missing.")
            continue
        if time_sampling is None or time_sampling <= 0:
            print(f"  Skipping record {record._id[:8]} for {record.peptide_name}: 'time_sampling' is missing or invalid ({time_sampling}).")
            continue

        full_data_filepath = os.path.join(data_path, data_file)

        try:
            # We assume the data_file is a CSV and contains data rows.
            # We need to count the number of data rows (excluding header).
            with open(full_data_filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None) # Read header
                row_count = sum(1 for row in reader) # Count data rows

            if row_count == 0:
                print(f"  Warning: Data file '{data_file}' at '{data_path}' has no data rows.")
                record_duration = 0.0
            else:
                # Calculate duration in seconds: number of data points / sampling frequency
                record_duration = row_count / time_sampling
                print(f"  Record ID: {record._id[:8]}, File: {data_file}, Path: {data_path}, "
                      f"Sampling: {time_sampling} Hz, Data Points: {row_count}, Duration: {record_duration:.4f} seconds")
            
            current_peptide_total_time += record_duration

        except FileNotFoundError:
            print(f"  ERROR: Data file not found for record {record._id[:8]}: {full_data_filepath}")
        except csv.Error as e:
            print(f"  ERROR: Could not read CSV file {full_data_filepath} for record {record._id[:8]}: {e}")
        except Exception as e:
            print(f"  ERROR: An unexpected error occurred processing {full_data_filepath} for record {record._id[:8]}: {e}")

    peptide_recording_times[peptide_name] = current_peptide_total_time
    total_recording_time_dataset += current_peptide_total_time

# --- Summary Table ---
print("\n\n--- Raw Recording Times Per Peptide ---")
print("{:<20} {:>15} {:>15}".format("Peptide Name", "Time (seconds)", "Time (hours)"))
print("-" * 50)
for peptide, duration_sec in peptide_recording_times.items():
    duration_hr = duration_sec / 3600
    print("{:<20} {:>15.4f} {:>15.4f}".format(peptide, duration_sec, duration_hr))

print("-" * 50)
total_recording_time_dataset_hr = total_recording_time_dataset / 3600
print("{:<20} {:>15.4f} {:>15.4f}".format("TOTAL Dataset", total_recording_time_dataset, total_recording_time_dataset_hr))
print("---------------------------------------")
