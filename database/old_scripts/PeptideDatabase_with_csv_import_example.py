import json
import os
import csv
from dataclasses import dataclass, asdict, field
from datetime import datetime
import uuid

# --- PeptideData Class Definition ---
@dataclass
class PeptideData:
    """
    A dataclass to store peptide translocation experiment data.
    """
    _id: str = field(init=False)  # Unique serialized ID
    date: str = field(init=False) # Date when the record was created

    peptide_name: str
    peptide_sequence: str
    nanopore_name: str
    user: str
    voltage: float
    buffer: str
    data_file: str
    time_sampling: float
    simulation: bool
    experimental: bool
    comments: str = "" # Optional field

    def __post_init__(self):
        """
        Initializes _id and date fields after the dataclass is created.
        """
        if not hasattr(self, '_id'): # Only generate if not already set (e.g., when loading from DB)
            self._id = str(uuid.uuid4()) # Generates a unique UUID for the ID
        if not hasattr(self, 'date'): # Only generate if not already set
            self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        with open(self.db_file, 'r') as f:
            return json.load(f)

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
        if query is None:
            return [PeptideData(**record) for record in records] # Return as PeptideData objects
        
        filtered_records = []
        for record in records:
            match = True
            for key, value in query.items():
                # Handle boolean conversion for query if needed
                if isinstance(value, bool) and isinstance(record.get(key), str):
                    if str(record.get(key)).lower() != str(value).lower():
                        match = False
                        break
                elif key not in record or record[key] != value:
                    match = False
                    break
            if match:
                filtered_records.append(PeptideData(**record))
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
                    if key in record:
                        record[key] = value
                    else:
                        print(f"Warning: Field '{key}' not found in record ID {record_id}.")
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
        """
        added_count = 0
        skipped_count = 0
        records = self._read_db()

        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Get the expected fields from PeptideData (excluding _id and date as they are auto-generated)
            peptide_data_fields = {f.name for f in PeptideData.__dataclass_fields__.values() 
                                   if f.name not in ['_id', 'date']}

            for i, row in enumerate(reader):
                processed_row = {}
                missing_fields = []
                type_conversion_errors = {}

                # Check for missing required fields
                for field_name in peptide_data_fields:
                    if field_name not in row or not row[field_name].strip():
                        # comments is optional, so we handle it separately
                        if field_name != 'comments':
                            missing_fields.append(field_name)
                        else: # Ensure comments field is present, even if empty
                            processed_row['comments'] = ""
                    else:
                        value = row[field_name].strip()
                        # Type conversion
                        try:
                            field_type = PeptideData.__dataclass_fields__[field_name].type
                            if field_type == float:
                                processed_row[field_name] = float(value)
                            elif field_type == bool:
                                processed_row[field_name] = value.lower() in ('true', '1', 't', 'y', 'yes')
                            else:
                                processed_row[field_name] = value
                        except ValueError:
                            type_conversion_errors[field_name] = value
                
                if missing_fields:
                    print(f"Skipping row {i+2} (CSV line {i+2}) due to missing required fields: {', '.join(missing_fields)}")
                    skipped_count += 1
                    continue
                
                if type_conversion_errors:
                    error_details = ", ".join([f"{field}: '{val}'" for field, val in type_conversion_errors.items()])
                    print(f"Skipping row {i+2} (CSV line {i+2}) due to type conversion errors for fields: {error_details}")
                    skipped_count += 1
                    continue

                try:
                    # Create a PeptideData instance. Let __post_init__ handle _id and date.
                    new_peptide_data = PeptideData(**processed_row)
                    records.append(asdict(new_peptide_data))
                    added_count += 1
                except Exception as e:
                    print(f"Skipping row {i+2} (CSV line {i+2}) due to an unexpected error during record creation: {e}")
                    skipped_count += 1
        
        self._write_db(records)
        print(f"\nCSV import complete. Added {added_count} records, skipped {skipped_count} records.")


# --- Example Usage ---
if __name__ == "__main__":
    # Clean up previous db for a fresh start in example
    if os.path.exists("peptide_data.json"):
        os.remove("peptide_data.json")
    
    db = PeptideDatabase()

    # Create a dummy CSV file for demonstration
    csv_content = """peptide_name,peptide_sequence,nanopore_name,user,voltage,buffer,data_file,time_sampling,simulation,experimental,comments
PeptideA,SEQ1,ONT-R9,Alice,100.0,PBS,data_A_001.bin,0.1,FALSE,TRUE,"First experimental run"
PeptideB,SEQ2,MinION,Bob,80.5,HEPES,data_B_001.csv,0.05,TRUE,FALSE,"Simulation with 0.1 std dev noise"
PeptideC,SEQ3,ONT-R10,Charlie,120.0,Tris,data_C_001.bin,0.2,FALSE,TRUE,"High voltage run"
PeptideD,SEQ4,MinION,Diana,75.0,Citrate,data_D_001.csv,0.02,TRUE,FALSE,"Another simulation"
PeptideE,,MinION,Eve,90.0,MES,data_E_001.csv,0.1,TRUE,FALSE, # Missing peptide_sequence
PeptideF,SEQ6,MinION,Frank,abc,HEPES,data_F_001.csv,0.1,TRUE,FALSE,"Invalid voltage"
"""
    with open("sample_peptide_data.csv", "w") as f:
        f.write(csv_content)

    print("--- Importing records from CSV ---")
    db.import_from_csv("sample_peptide_data.csv")

    print("\n--- Verifying imported records ---")
    all_peptides = db.retrieve_records()
    for p in all_peptides:
        print(f"  ID: {p._id}, Name: {p.peptide_name}, User: {p.user}, File: {p.data_file}, Simulation: {p.simulation}")

    # Clean up the dummy CSV file
    if os.path.exists("sample_peptide_data.csv"):
        os.remove("sample_peptide_data.csv")

    # Example of adding a single record manually after CSV import
    print("\n--- Adding a single record manually ---")
    peptide_manual = PeptideData(
        peptide_name="PeptideManual",
        peptide_sequence="MANUALSEQ",
        nanopore_name="LabPore",
        user="Grace",
        voltage=95.0,
        buffer="BufferX",
        data_file="manual_test.dat",
        time_sampling=0.15,
        simulation=True,
        experimental=False,
        comments="Added manually after CSV import."
    )
    db.add_record(peptide_manual)

    print("\n--- Retrieving all records after manual add ---")
    all_peptides_after_manual = db.retrieve_records()
    for p in all_peptides_after_manual:
        print(f"  ID: {p._id}, Name: {p.peptide_name}, User: {p.user}")
