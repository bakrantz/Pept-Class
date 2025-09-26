#!/usr/bin/perl -w
#
# This script automates the uploading of all 20 natural amino acid PDB files
# to the GETAREA server and saves the output.
#
# It incorporates a 10-second delay between requests to avoid server timeouts.
#
# Written by Surendra Negi, UTMB Galveston.
# Refactored by Gemini.
#
use strict;
use LWP;
use LWP::UserAgent;
use HTTP::Request::Common;
use HTTP::Headers;
use Time::HiRes 'sleep';

# for ssl connection
use IO::Socket::SSL qw( SSL_VERIFY_NONE );

$ENV{PERL_LWP_SSL_VERIFY_HOSTNAME} = 0;
$ENV{HTTPS_DEBUG} = 1;

my @all_PDB_files = ("Ala.pdb", "Arg.pdb", "Asn.pdb", "Asp.pdb", "Cys.pdb", "Glu.pdb", "Gln.pdb", "Gly.pdb", "His.pdb", "Ile.pdb", "Leu.pdb", "Lys.pdb", "Met.pdb", "Phe.pdb", "Pro.pdb", "Ser.pdb", "Thr.pdb", "Trp.pdb", "Tyr.pdb", "Val.pdb");

my $url = "https://curie.utmb.edu/cgi-bin/getarea.cgi";
my $name = "test";
my $email = "bkrantz\@umaryland.edu";
my $probesize = "1.4"; # Water probe (1.4 angstroms)
my $gradi = "n"; # (y OR n) if you are interested in gradient calculation
my $output = "4"; # 1= Total Area/Energy, 2 = Area per residue, 3 = Area per atom type, 4 = Area per individual atom type.

my $ua = LWP::UserAgent->new(
    ssl_opts => {
        verify_hostname => 0,
        SSL_verify_mode => SSL_VERIFY_NONE,
    }
);

# Set a browser-like user agent
$ua->agent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36');

# Loop through each PDB file and submit it
foreach my $pdb_file (@all_PDB_files) {
    print "Submitting $pdb_file to GETAREA...\n";

    # Create the HTTP POST request with the form data
    my $request = HTTP::Request::Common::POST(
        $url,
        Content_Type => 'form-data',
        Content => [
            'water' => $probesize,
            'gradient' => $gradi,
            'name' => $name,
            'email' => $email,
            'Method' => $output,
            'PDBfile' => ["$pdb_file"]
        ],
    );

    # Send the request and get the response
    my $response = $ua->request($request);

    # Check the response for success
    if ($response->is_success) {
        my $html = $response->content;
        my $output_file = $pdb_file . ".txt";
        
        # Open a new file and write the HTML content
        open(my $fh, '>', $output_file) or die "Cannot open file '$output_file' for writing: $!";
        print $fh $html;
        close($fh);
        
        print "Successfully saved output to '$output_file'.\n";
    } else {
        print "Failed to submit $pdb_file: " . $response->status_line . "\n";
    }

    # Pause for 10 seconds to avoid server issues
    print "Pausing for 10 seconds...\n\n";
    sleep(10);
}

print "Script finished. All files processed.\n";

exit;
