import json
import os
import csv
from dataclasses import dataclass, asdict, field
from datetime import datetime
import uuid
from typing import Optional, Any # Import Optional for optional fields

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
    data_file: str
    user: str
    simulation: bool
    experimental: bool

    # Optional fields with default None or empty string
    peptide_sequence: Optional[str] = None
    nanopore_name: Optional[str] = None
    voltage: Optional[float] = None
    buffer: Optional[str] = None
    time_sampling: Optional[float] = None
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
        # Filter out fields that are not part of __init__ for the dataclass
        # and also handle optional fields that might be None
        init_args = {}
        for f in cls.__dataclass_fields__.values():
            if f.init: # Only include fields that are part of the __init__ method
                if f.name in data:
                    init_args[f.name] = data[f.name]
                elif f.default is not field.MISSING: # Use default if not present
                    init_args[f.name] = f.default
                elif f.default_factory is not field.MISSING: # Use default_factory if not present
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
    def __init__(self, db_file="peptide_data.json"):
        self.db_file = db_file
        self._initialize_db()

    def _initialize_db(self):
        """
        Creates the database file if it doesn't exist.
        """
        if not os.path.exists(self.db_file):
            with open(self.db_file, 'w') as f:
                json.dump([], f) # Initialize with an empty list

    def _read_db(self):
        """
        Reads all records from the database file.
        """
        try:
            with open(self.db_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: '{self.db_file}' is empty or corrupted. Reinitializing.")
            self._write_db([]) # Reinitialize with an empty list
            return []


    def _write_db(self, data):
        """
        Writes data to the database file.
        """
        with open(self.db_file, 'w') as f:
            json.dump(data, f, indent=4) # Use indent for pretty printing JSON

    def add_record(self, peptide_data: PeptideData):
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
        If no query is provided, all records are returned.
        Query format: {"field_name": "value"}
        """
        records = self._read_db()
        
        # Use the new from_dict class method to safely reconstruct PeptideData objects
        if query is None:
            return [PeptideData.from_dict(record) for record in records]
        
        filtered_records = []
        for record_dict in records: # Renamed to record_dict to avoid confusion
            match = True
            for key, value in query.items():
                if key not in record_dict:
                    match = False
                    break
                
                record_value = record_dict[key]
                
                # Handle boolean conversion for query if needed (e.g., query for "True" vs True)
                if isinstance(value, bool) and isinstance(record_value, str):
                    if str(record_value).lower() != str(value).lower():
                        match = False
                        break
                elif record_value != value:
                    match = False
                    break
            if match:
                filtered_records.append(PeptideData.from_dict(record_dict)) # Use from_dict
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
                # Attempt to convert update values to target types if possible
                for key, value in updates.items():
                    if key in PeptideData.__dataclass_fields__:
                        field_type = PeptideData.__dataclass_fields__[key].type
                        try:
                            if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                                # Get the actual type if it's Optional[T]
                                actual_type = field_type.__args__[0]
                            else:
                                actual_type = field_type

                            if actual_type == float:
                                record[key] = float(value)
                            elif actual_type == bool:
                                record[key] = str(value).lower() in ('true', '1', 't', 'y', 'yes')
                            else:
                                record[key] = value
                        except ValueError:
                            print(f"Warning: Could not convert update value '{value}' for field '{key}' to expected type {actual_type}. Skipping update for this field.")
                            continue # Skip to next update key
                    else:
                        print(f"Warning: Field '{key}' not found in record ID {record_id} or PeptideData definition.")
                        continue # Skip to next update key
                
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
            
            # Get the expected fields from PeptideData (excluding _id and date as they are auto-generated)
            # and their types
            peptide_data_field_types = {f.name: f.type for f in PeptideData.__dataclass_fields__.values() 
                                        if f.name not in ['_id', 'date']}

            for i, row in enumerate(reader):
                processed_row = {}
                type_conversion_errors = {}
                
                # Iterate through all fields defined in PeptideData to ensure proper handling
                for field_name, field_type in peptide_data_field_types.items():
                    value_from_csv = row.get(field_name, "").strip() # Use .get() to handle missing columns gracefully

                    # If the field is Optional[T], get the inner type T
                    if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                        actual_type = field_type.__args__[0] if field_type.__args__ else Any
                    else:
                        actual_type = field_type

                    # Handle empty strings for optional fields -> convert to None
                    if value_from_csv == "" and actual_type in (str, float, bool) and field_name != 'comments':
                        processed_row[field_name] = None
                        continue # Move to next field

                    # Type conversion for non-empty values
                    try:
                        if actual_type == float:
                            processed_row[field_name] = float(value_from_csv)
                        elif actual_type == bool:
                            processed_row[field_name] = value_from_csv.lower() in ('true', '1', 't', 'y', 'yes')
                        else: # String or other types, just assign directly
                            processed_row[field_name] = value_from_csv
                    except ValueError:
                        type_conversion_errors[field_name] = value_from_csv

                if type_conversion_errors:
                    error_details = ", ".join([f"{field}: '{val}'" for field, val in type_conversion_errors.items()])
                    print(f"Skipping row {i+2} (CSV line {i+2}) due to type conversion errors for fields: {error_details}")
                    skipped_count += 1
                    continue

                try:
                    # Create a PeptideData instance using the processed_row.
                    # from_dict handles the _id and date fields being init=False
                    new_peptide_data = PeptideData.from_dict(processed_row)
                    records.append(asdict(new_peptide_data))
                    added_count += 1
                except Exception as e:
                    print(f"Skipping row {i+2} (CSV line {i+2}) due to an unexpected error during record creation: {e} (Row data: {processed_row})")
                    skipped_count += 1
        
        self._write_db(records)
        print(f"\nCSV import complete. Added {added_count} records, skipped {skipped_count} records.")


# --- Example Usage ---
if __name__ == "__main__":
    db = PeptideDatabase()

    print("--- Importing records from CSV ---")
    db.import_from_csv("PeptideDatabase.csv")

    print("\n--- Verifying imported records ---")
    all_peptides = db.retrieve_records()
    for p in all_peptides:
        print(f"  ID: {p._id}, Name: {p.peptide_name}, User: {p.user}, File: {p.data_file}, Simulation: {p.simulation}")
