use lib ("$ENV{UAL_RHIC}/api/", "$ENV{UAL_SXF}/api");

my $file = "rhic";

use UAL::SXF::Parser;

# ###########################################################
# Read a SXF file
# ###########################################################

print "read the SXF file  ", time, "\n";
my $sxf_parser = new UAL::SXF::Parser();
$sxf_parser->read("in/$file" . ".sxf", "echo/$file" . ".sxf");
print "end                ", time, "\n";

# ###########################################################
# Write a SXF file
# ###########################################################

print "write the SXF file ", time, "\n";
$sxf_parser->write("out/$file" . ".sxf");
print "end                ", time, "\n";

# ###########################################################
# Track partcile
# ###########################################################

use RHIC::SMF::SMF;
use RHIC::UI::Shell;

# Make the shell
my $shell = new RHIC::UI::Shell();

# Define DA Space
my $maxOrder = 7;
$shell->space($maxOrder);

# Select an accelerator  for operations
$shell->use("blue");

# Define beam parameters
$shell->beam(energy => 250.);

# Define initial particle coordinates
$shell->start([1.0e-3, 0.0, 1.0e-3, 0.0, 0.0, 1.0e-3]);

# Track particles 
$shell->firstturn("print" => "./out/firstturn/$file" . ".sxf", "observe" => ""); 


