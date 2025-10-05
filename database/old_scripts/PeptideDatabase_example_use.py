import json
import os
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
        self._id = str(uuid.uuid4()) # Generates a unique UUID for the ID
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
                if key not in record or record[key] != value:
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

# --- Example Usage ---
if __name__ == "__main__":
    db = PeptideDatabase()

    # --- Add Records ---
    print("--- Adding Records ---")
    peptide1 = PeptideData(
        peptide_name="PeptideA",
        peptide_sequence="AGCTAGCTAG",
        nanopore_name="ONT-R9",
        user="Alice",
        voltage=100.0,
        buffer="PBS",
        data_file="exp_A_001.bin",
        time_sampling=0.1,
        simulation=False,
        experimental=True,
        comments="First experimental run with PeptideA"
    )
    id1 = db.add_record(peptide1)

    peptide2 = PeptideData(
        peptide_name="PeptideB",
        peptide_sequence="GCTAGCTAGC",
        nanopore_name="MinION",
        user="Bob",
        voltage=80.0,
        buffer="HEPES",
        data_file="sim_B_001.csv",
        time_sampling=0.05,
        simulation=True,
        experimental=False,
        comments="Simulation data for PeptideB"
    )
    id2 = db.add_record(peptide2)

    peptide3 = PeptideData(
        peptide_name="PeptideA",
        peptide_sequence="AGCTAGCTAG",
        nanopore_name="ONT-R10",
        user="Alice",
        voltage=110.0,
        buffer="PBS",
        data_file="exp_A_002.bin",
        time_sampling=0.1,
        simulation=False,
        experimental=True,
        comments="Second experimental run for PeptideA, different nanopore"
    )
    id3 = db.add_record(peptide3)

    print("\n--- Retrieving Records ---")
    # Retrieve all records
    print("All records:")
    all_peptides = db.retrieve_records()
    for p in all_peptides:
        print(f"  ID: {p._id}, Name: {p.peptide_name}, User: {p.user}, Date: {p.date}")

    print("\nRecords for PeptideA:")
    # Retrieve records by peptide_name
    peptide_a_records = db.retrieve_records(query={"peptide_name": "PeptideA"})
    for p in peptide_a_records:
        print(f"  ID: {p._id}, Name: {p.peptide_name}, User: {p.user}, Data File: {p.data_file}")

    print("\nSimulation data:")
    # Retrieve simulation data
    sim_data = db.retrieve_records(query={"simulation": True})
    for p in sim_data:
        print(f"  ID: {p._id}, Name: {p.peptide_name}, Data File: {p.data_file}")

    print("\n--- Editing Records ---")
    # Edit a record
    print(f"Original comments for {id1}:")
    for p in db.retrieve_records(query={"_id": id1}):
        print(f"  Comments: {p.comments}")

    db.edit_record(id1, {"comments": "Updated comments: This run had excellent signal-to-noise."})
    db.edit_record(id1, {"new_field": "test"}) # Example of adding a non-existent field (will print warning)

    print(f"Updated comments for {id1}:")
    for p in db.retrieve_records(query={"_id": id1}):
        print(f"  Comments: {p.comments}")

    print("\nAttempting to edit a non-existent record:")
    db.edit_record("non-existent-id", {"comments": "This won't work"})

    # You can inspect the 'peptide_data.json' file in the same directory as your script
    # to see the stored data.
