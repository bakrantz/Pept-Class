# Database structure

There are two modules that have classes designed to access databases.
	(1) PeptideDatabase.py
	(2) PeptideEventsDatabase.py

The first module accesses meta data and filenames and paths to the raw data (.atf files) and conductance state labeled raw data (.csv files). You will need to update the paths for your filesystem on your computer. There is an update file paths method in PeptideDatabase.py, which will search the entire project root that you specify, and it will find the paths to the raw files in the json database file (called peptide_data.json). I would not delete peptide_data.json since it is mostly correct--just the paths are not going to be correct on your system.

The second module accesses meta data and filenames and paths to the processed translocation events .pkl files. These .pkl files are cached to prevent processing the same raw files repeatedly. You likely want to clear the main database file in peptide_events_data.json since the filepaths will be wrong on your computer. Just delete this json file and it will be recreated with appropriate paths. 

## Raw data

Because the raw data is too large to maintain on Github. You will need to download this data from Zenodo. There are different conductance state labelings of the raw data, either by Clampfit or a custom K-Means clustering approach. I recommend unzipping these archives into the raw_data directory in the database directory from these Zenodo links.

Earlier work on wild-type PA nanopore used Clampfit:
	- https://doi.org/10.5281/zenodo.16983789

Later work on PA F427A and PA F427Y used K-Means routine:
	- https://doi.org/10.5281/zenodo.16985194

Again later wild-type PA was labeled for conductance state with K-Means routine:
	- https://doi.org/10.5281/zenodo.17143408

Full analysis of all 20 canonical amino acids in guest-host peptide background for wild-type PA using K-Means labeling routine:
	- https://doi.org/10.5281/zenodo.21453990
