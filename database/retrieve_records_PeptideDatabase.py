from PeptideDatabase import PeptideData, PeptideDatabase # Assuming PeptideDatabase.py is in your Python path

db = PeptideDatabase()

print("\n--- Records in Database ---")
# It's good practice to get all records just once if you need to iterate/display them all
all_peptides = db.retrieve_records()
for p in all_peptides:
    # You might want to format this for clarity
    print(f"  ID: {p._id[:8]}..., Name: {p.peptide_name}, File: {p.data_file}, Nanopore Name: {p.nanopore_name}, Conc: {p.peptide_conc} nM, Date: {p.date}")
print(f"Total records: {len(all_peptides)}")

peptide_names = ["guesthost_Ala", "guesthost_Leu", "guesthost_Phe", "guesthost_Thr", "guesthost_Trp", "guesthost_TrpDL", "guesthost_Tyr"]

print("\n--- Find peptides in Database at specified range of peptide concentrations ---")
conc_low, conc_high = (5, 20)
print(f"Concentration range from {conc_low} nM to {conc_high} nM.")

for peptide_name in peptide_names:
    # Construct the query dictionary
    # This single query will filter by both peptide_name AND the concentration range
    peptide_query = {
        'experimental': True,
        'nanopore_name': 'PA_F427Y',
        'voltage': 70,
        'peptide_name': peptide_name,
        'time_sampling': 400,
        'peptide_conc': {'$gte': conc_low, '$lte': conc_high}
    }
    
    # Call retrieve_records with the enhanced query
    result_peptide_records = db.retrieve_records(peptide_query)
    
    count = len(result_peptide_records)
    print(f"{peptide_name} has {count} records in concentration range.")

    # Optional: Print details of the found records
    # if count > 0:
    #     print(f"  Details for {peptide_name}:")
    #     for p in result_peptide_records:
    #         print(f"    ID: {p._id[:8]}..., Conc: {p.peptide_conc} nM, File: {p.data_file}")
