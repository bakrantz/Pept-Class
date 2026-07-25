from PeptideDatabase import PeptideData, PeptideDatabase

db = PeptideDatabase()

print("--- Importing records from CSV ---")

# db.import_from_csv("PeptideDatabase-simulation_peptides_A-J.csv")

# db.import_from_csv("PeptideDatabase-70_mv_guesthost_peptides.csv")

# db.import_from_csv("PeptideDatabase-70_mv_F427Y_guesthost_peptides.csv")

# db.import_from_csv("PeptideDatabase-70_mv_F427A_guesthost_peptides.csv")

db.import_from_csv("PeptideDatabase-other_mv_guesthost_peptides.csv")

# Jaylen's path
# db.update_data_paths(r"C:\Users\jayle\source\repos\exp_pep_class\database")

# Bryan's path
db.update_data_paths("/Users/bakrantz/Documents/python/database")

print("\n--- Records after import ---")
all_peptides = db.retrieve_records()
for p in all_peptides:
    print(f"  ID: {p._id}, Name: {p.peptide_name}, File: {p.data_file}, Date: {p.date}")
print(f"Total records: {len(all_peptides)}")
