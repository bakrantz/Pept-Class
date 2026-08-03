#!/usr/bin/perl -w

# This script processes the HTML output files from the GETAREA server,
# extracts per-atom SASA values, sums them by atom type, calculates
# a total solvation energy, and writes the results to a CSV file.
#
# It requires the output files to be named in the format {3-letter-code}.pdb.txt
# (e.g., "Ala.pdb.txt", "Trp.pdb.txt").

use strict;
use warnings;
use File::Basename;

# --- Atom Solvation Parameters (in cal/mol/Angstrom^2) ---
# These values are based on the internet searches.
my %solvation_params_old = (
    'C_aliphatic_area'   => 19,
    'C_aromatic_area'    => 7,
    'N_non_charged_area' => -7,
    'N_charged_area'     => -87,
    'O_non_charged_area' => -47,
    'O_charged_area'     => -110,
    'S_area'             => -1,
);

# These ASP values are the empirically derived set from Ooi et al. (1987) PNAS 84, 3086-3090.
# This is a widely accepted standard for SASA-based solvation energy calculations.
my %solvation_params = (
    'C_aliphatic_area'   => 18,     # Aliphatic carbon
    'C_aromatic_area'    => 8,      # Aromatic carbon
    'N_non_charged_area' => -48,    # Uncharged nitrogen (e.g., amide)
    'N_charged_area'     => -113,   # Charged nitrogen (e.g., Lys, Arg)
    'O_non_charged_area' => -58,    # Uncharged oxygen (e.g., carbonyl, hydroxyl)
    'O_charged_area'     => -106,   # Charged oxygen (e.g., carboxylate)
    'S_area'             => 21,     # Sulfur
);

# --- Atom Type Classification ---
# This function maps PDB atom names to a solvation category.
# It is based on standard PDB conventions for natural amino acids.
sub classify_atom {
    my ($residue_code, $atom_name) = @_;

    # All backbone atoms are considered non-charged
    if ($atom_name eq 'N' || $atom_name eq 'O') {
        return "N_non_charged_area" if $atom_name eq 'N';
        return "O_non_charged_area" if $atom_name eq 'O';
    }

    # All backbone carbons are aliphatic
    if ($atom_name eq 'CA' || $atom_name eq 'C') {
        return "C_aliphatic_area";
    }

    # --- Side-chain atom classification ---
    # S atoms
    if ($atom_name =~ /^S/) {
        return "S_area";
    }

    # Charged N atoms (by convention, one atom in a resonant group is assigned the charge)
    if (($residue_code eq 'LYS' && $atom_name eq 'NZ') ||
        ($residue_code eq 'ARG' && $atom_name eq 'NH1') ||
        ($residue_code eq 'HIS' && $atom_name eq 'ND1')) {
        return "N_charged_area";
    }
    
    # Charged O atoms (by convention, one of the two carboxylate oxygens is charged)
    if (($residue_code eq 'ASP' && $atom_name eq 'OD1') ||
        ($residue_code eq 'GLU' && $atom_name eq 'OE1')) {
        return "O_charged_area";
    }

    # Aromatic C atoms
    if (($residue_code eq 'PHE' && $atom_name =~ /^(CG|CD|CE|CZ)/) ||
        ($residue_code eq 'TYR' && $atom_name =~ /^(CG|CD|CE|CZ)/) ||
        ($residue_code eq 'TRP' && $atom_name =~ /^(CG|CD|CE|CZ|CH)/) ||
        ($residue_code eq 'HIS' && $atom_name =~ /^(CG|CD|CE)/)) {
        return "C_aromatic_area";
    }

    # Non-charged N atoms (side-chain)
    if (($residue_code eq 'ASN' && $atom_name =~ /^ND/) ||
        ($residue_code eq 'GLN' && $atom_name =~ /^NE/) ||
        ($residue_code eq 'TRP' && $atom_name eq 'NE1') ||
        ($residue_code eq 'ARG' && ($atom_name eq 'NE' || $atom_name eq 'NH2')) ||
        ($residue_code eq 'HIS' && $atom_name eq 'NE2')) {
        return "N_non_charged_area";
    }

    # Non-charged O atoms (side-chain)
    if (($residue_code eq 'SER' && $atom_name eq 'OG') ||
        ($residue_code eq 'THR' && $atom_name eq 'OG1') ||
        ($residue_code eq 'TYR' && $atom_name eq 'OH') ||
        ($residue_code eq 'ASN' && $atom_name eq 'OD1') ||
        ($residue_code eq 'GLN' && $atom_name eq 'OE1') ||
        ($residue_code eq 'ASP' && $atom_name eq 'OD2') ||
        ($residue_code eq 'GLU' && $atom_name eq 'OE2')) {
        return "O_non_charged_area";
    }

    # All other carbons are considered aliphatic
    if ($atom_name =~ /^C/) {
        return "C_aliphatic_area";
    }

    # If an atom is not classified, it is ignored
    return undef;
}

# --- Main Script ---

my @amino_acids = ("Ala", "Arg", "Asn", "Asp", "Cys", "Glu", "Gln", "Gly", "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val");

# Define the output CSV file and header
my $output_csv = "solvation_summary.csv";
open(my $csv_fh, '>', $output_csv) or die "Cannot open file '$output_csv' for writing: $!";
print $csv_fh "Residue 3-letter name, C_aliphatic_area, C_aromatic_area, N_non_charged_area, N_charged_area, O_non_charged_area, O_charged_area, S_area, total_energy\n";

print "Starting analysis of GETAREA output files...\n";

foreach my $aa (@amino_acids) {
    my $input_file = "$aa.pdb.txt";
    
    unless (-e $input_file) {
        warn "Warning: Input file '$input_file' not found. Skipping.\n";
        next;
    }

    # Initialize area accumulators for each residue
    my %areas = (
        'C_aliphatic_area'   => 0,
        'C_aromatic_area'    => 0,
        'N_non_charged_area' => 0,
        'N_charged_area'     => 0,
        'O_non_charged_area' => 0,
        'O_charged_area'     => 0,
        'S_area'             => 0,
    );
    
    print "Processing $input_file...\n";
    
    my $content;
    {
        local $/; # Enable "slurp" mode to read the whole file at once
        open(my $file_fh, '<', $input_file) or die "Cannot open '$input_file': $!";
        $content = <$file_fh>;
        close($file_fh);
    }
    
    # Regex to find the block of interest
    my ($data_block) = $content =~ m/<td><pre>\s*ATOM\s+NAME\s+RESIDUE\s+AREA\/ENERGY<\/pre><\/td>(.*?)<td><pre>\s*-{20,}<\/pre><\/td>/s;
    
    if (!$data_block) {
        warn "Warning: No data block found in '$input_file'. Skipping.\n";
        next;
    }

    # Clean up the data block and split into lines
    my @lines = split /\n/, $data_block;
    
    # Process each line to extract data
    foreach my $line (@lines) {
        my ($atom_name, $residue_name, $area);
	
        # Regex to match the data format
        if ($line =~ /\s*\d+\s+([A-Z0-9]+)\s+([A-Z]+)\s+\d+\s+([\d\.]+)/) {
            $atom_name = $1;
            $residue_name = $2;
            $area = $3;
	    
            # Use the classifier to get the atom type
            my $type = classify_atom($residue_name, $atom_name);
            
            # Sum the area if a valid type is found
            if (defined $type) {
                $areas{$type} += $area;
            } else {
                warn "Warning: Could not classify atom '$atom_name' in residue '$residue_name'.\n";
            }
        }
    }

    # Check if any SASA values were found and provide feedback
    my $total_sasa = 0;
    foreach my $type (keys %areas) {
        $total_sasa += $areas{$type};
    }

    if ($total_sasa == 0) {
        warn "Warning: No SASA values were extracted from '$input_file'. Please check the file's content.\n";
    }

    # Calculate total energy
    my $total_energy = 0;
    foreach my $type (keys %areas) {
        $total_energy += $areas{$type} * $solvation_params{$type};
    }

    # Write the results to the CSV file
    printf $csv_fh "%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
        $aa,
        $areas{'C_aliphatic_area'},
        $areas{'C_aromatic_area'},
        $areas{'N_non_charged_area'},
        $areas{'N_charged_area'},
        $areas{'O_non_charged_area'},
        $areas{'O_charged_area'},
        $areas{'S_area'},
        $total_energy;
}

close($csv_fh);
print "Analysis complete. Summary saved to '$output_csv'.\n";

exit;
