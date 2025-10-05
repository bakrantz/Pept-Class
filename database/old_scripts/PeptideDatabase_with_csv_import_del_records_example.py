import json
import os
import csv
from dataclasses import dataclass, asdict, field
from datetime import datetime
import uuid
from typing import Optional, Any, List, Union # Union is now correctly imported

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

    def delete_records(self, record_ids: Union[str, List[str]]):
        """
        Deletes one or more records from the database based on their _id(s).
        Takes either a single _id string or a list of _id strings.
        """
        if isinstance(record_ids, str):
            record_ids = [record_ids] # Convert single ID to a list for consistent processing

        records = self._read_db()
        original_count = len(records)
        
        # Keep only records whose _id is NOT in the list of IDs to delete
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
            
            peptide_data_field_types = {f.name: f.type for f in PeptideData.__dataclass_fields__.values() 
                                        if f.name not in ['_id', 'date']}

            for i, row in enumerate(reader):
                processed_row = {}
                type_conversion_errors = {}
                
                for field_name, field_type in peptide_data_field_types.items():
                    value_from_csv = row.get(field_name, "").strip()

                    if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                        actual_type = field_type.__args__[0] if field_type.__args__ else Any
                    else:
                        actual_type = field_type

                    if value_from_csv == "" and actual_type in (str, float, bool) and field_name != 'comments':
                        processed_row[field_name] = None
                        continue

                    try:
                        if actual_type == float:
                            processed_row[field_name] = float(value_from_csv)
                        elif actual_type == bool:
                            processed_row[field_name] = value_from_csv.lower() in ('true', '1', 't', 'y', 'yes')
                        else:
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
        Two records are considered duplicates if all their fields (excluding '_id' and 'date')
        are identical.
        """
        records = self._read_db()
        unique_records = {}
        ids_to_delete = []

        # List of fields to ignore when checking for duplicates
        # ADD 'comments' HERE if you don't want it to affect duplication logic
        ignored_fields = {'_id', 'date', 'comments'}

        # Get all field names from PeptideData to ensure we compare all relevant fields
        all_peptide_fields = {f.name for f in PeptideData.__dataclass_fields__.values()}
        fields_to_compare = sorted(list(all_peptide_fields - ignored_fields))

        print("\n--- Checking for Duplicate Records ---")

        for record_dict in records:
            # Create a tuple of relevant field values for comparison
            comparison_key_values = []
            for field_name in fields_to_compare:
                # Use .get() to safely retrieve values, defaulting to None if key is missing
                # This ensures consistent comparison for optional fields.
                comparison_key_values.append(record_dict.get(field_name))
            
            comparison_key = tuple(comparison_key_values)
            
            # Convert date strings to datetime objects for proper comparison
            current_date_str = record_dict.get('date')
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S") if current_date_str else datetime.min # Use min date if date is missing

            if comparison_key in unique_records:
                # Found a potential duplicate
                existing_record = unique_records[comparison_key]
                existing_date = datetime.strptime(existing_record['date'], "%Y-%m-%d %H:%M:%S")
                
                if current_date < existing_date:
                    # Current record is older, keep it and mark the previously stored one for deletion
                    ids_to_delete.append(existing_record['_id'])
                    unique_records[comparison_key] = record_dict # Update with the older record
                else: # This covers current_date >= existing_date
                    # Existing record is older or same age, mark the current one for deletion
                    ids_to_delete.append(record_dict['_id'])
            else:
                # First time seeing this unique combination of fields
                unique_records[comparison_key] = record_dict
        
        if ids_to_delete:
            self.delete_records(ids_to_delete)
        else:
            print("No duplicate records found.")

# --- Example Usage ---
if __name__ == "__main__":
    # Clean up previous db for a fresh start in example
    if os.path.exists("peptide_data.json"):
        os.remove("peptide_data.json")
    
    db = PeptideDatabase()

    # Create a dummy CSV file for demonstration with duplicates
    # Note: 'buffer' is empty for simulation records now.
    # 'peptide_sequence' is empty for PeptideE.
    # 'voltage' is 'abc' for PeptideF (will cause conversion error in a real scenario, but handled here).
    csv_content = """peptide_name,peptide_sequence,nanopore_name,user,voltage,buffer,data_file,time_sampling,simulation,experimental,comments
PeptideA,SEQ1,ONT-R9,Alice,100.0,PBS,data_A_001.bin,0.1,FALSE,TRUE,"First experimental run"
PeptideB,SEQ2,MinION,Bob,80.5,,data_B_001.csv,0.05,TRUE,FALSE,"Simulation with 0.1 std dev noise (buffer is empty)"
PeptideC,SEQ3,ONT-R10,Charlie,120.0,Tris,data_C_001.bin,0.2,FALSE,TRUE,"High voltage run"
PeptideD,SEQ4,MinION,Diana,75.0,,data_D_001.csv,0.02,TRUE,FALSE,"Another simulation (buffer is empty)"
PeptideA,SEQ1,ONT-R9,Alice,100.0,PBS,data_A_001.bin,0.1,FALSE,TRUE,"Duplicate of PeptideA, created later"
PeptideB,SEQ2,MinION,Bob,80.5,,data_B_001.csv,0.05,TRUE,FALSE,"Another duplicate simulation (buffer is empty)"
PeptideX,SEQX,ONT-R9,Alice,100.0,PBS,data_X_001.bin,0.1,FALSE,TRUE,"A unique record for X"
PeptideB,SEQ2,MinION,Bob,80.5,,data_B_001.csv,0.05,TRUE,FALSE,"Third duplicate simulation (buffer is empty)"
PeptideA,SEQ1,ONT-R9,Alice,100.0,PBS,data_A_001.bin,0.1,FALSE,TRUE,"Third duplicate of PeptideA, oldest (will be kept)"
"""
    with open("sample_peptide_data.csv", "w") as f:
        f.write(csv_content)

    print("--- Importing records from CSV (with duplicates) ---")
    db.import_from_csv("sample_peptide_data.csv")

    print("\n--- Records before duplicate deletion ---")
    all_peptides_before_delete = db.retrieve_records()
    for p in all_peptides_before_delete:
        print(f"  ID: {p._id}, Name: {p.peptide_name}, File: {p.data_file}, Date: {p.date}")
    print(f"Total records: {len(all_peptides_before_delete)}")

    # --- Test delete_duplicate_records ---
    db.delete_duplicate_records()

    print("\n--- Records after duplicate deletion ---")
    all_peptides_after_delete = db.retrieve_records()
    for p in all_peptides_after_delete:
        print(f"  ID: {p._id}, Name: {p.peptide_name}, File: {p.data_file}, Date: {p.date}")
    print(f"Total records: {len(all_peptides_after_delete)}")

    # --- Test delete_records with a specific ID ---
    # Add a unique record to delete manually later
    single_record_to_delete = PeptideData(
        peptide_name="ToBeDeleted",
        data_file="delete_me.dat",
        user="TestUser",
        simulation=False,
        experimental=True,
        comments="This record is for testing deletion."
    )
    delete_id = db.add_record(single_record_to_delete)
    print(f"\nAdded record for deletion: ID {delete_id}")

    print("\n--- Deleting a specific record ---")
    db.delete_records(delete_id)

    print("\n--- Records after single record deletion ---")
    all_peptides_final = db.retrieve_records()
    for p in all_peptides_final:
        print(f"  ID: {p._id}, Name: {p.peptide_name}, File: {p.data_file}, Date: {p.date}")
    print(f"Total records: {len(all_peptides_final)}")

    # Clean up the dummy CSV file
    if os.path.exists("sample_peptide_data.csv"):
        os.remove("sample_peptide_data.csv")
