use lib ("$ENV{UAL_RHIC}/api/", "$ENV{UAL_SXF}/api");

use RHIC::SMF::SMF;
use RHIC::UI::Shell;

my $file = "rhic";
my $dir = "$ENV{UAL_RHIC}/data/rf";

# ###########################################################
# Initialize SMF
# ###########################################################

local $smf = new Pac::Smf();

require "$dir/rhicSMF_level_1.pl";
require "$dir/rhicSMF_level_2.pl";
require "$dir/rhicSMF_level_3.pl";

# Make the shell
my $shell = new RHIC::UI::Shell();

# Define DA Space
my $maxOrder = 7;
$shell->space($maxOrder);

# Read MAD input files
$shell->read("line"   => "blue",
	     "fields" => [
                         "$dir/rhicSMF_D5_deviations.pl",
			 "$dir/rhicSMF_D96_deviations.pl",
                         "$dir/rhicSMF_DRG_DR8_deviations.pl",
                         "$dir/rhicSMF_DRX_deviations.pl",
                         "$dir/rhicSMF_DRZ_deviations.pl",
                         "$dir/rhicSMF_QR4_deviations.pl",
                         "$dir/rhicSMF_QR7_deviations.pl",
                         "$dir/rhicSMF_QRG_deviations.pl",
                         "$dir/rhicSMF_QRI_deviations.pl",
                         "$dir/rhicSMF_QRJ_deviations.pl",
                         "$dir/rhicSMF_QRK_deviations.pl",
                         "$dir/rhicSMF_SRE_deviations.pl",
			  ],
);

# ###########################################################
# Track particle
# ###########################################################

# Select an accelerator  for operations
$shell->use("blue");

# Define beam parameters
$shell->beam(energy => 250.);

# Define initial particle coordinates
$shell->start([1.0e-3, 0.0, 1.0e-3, 0.0, 0.0, 1.0e-3]);

# Track particles 
$shell->firstturn("print" => "firstturn.1" . ".out", "observe" => ""); 

# ###########################################################
# Write a SXF file
# ###########################################################

print "store into the SXF format   ", time, "\n";

use UAL::SXF::Parser;
my $sxf_parser = new UAL::SXF::Parser();
$sxf_parser->write("$file" . ".in.sxf");

print "end                         ", time, "\n";
