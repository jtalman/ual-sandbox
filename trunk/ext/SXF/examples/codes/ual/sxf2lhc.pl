use lib ("$ENV{UAL_RHIC}/api/", "$ENV{UAL_SXF}/api");

use Carp;
use File::Basename;

my $file    = "lhc.v5.0";
my $dir     = "./in";
my $suffix  = "sxf";
my $lattice = "LHC";

# $dir = "./out";

use UAL::SXF::Parser;

# #############################################################
# Read a SXF file
# #############################################################

print "read the SXF file  ", time, "\n";
my $sxf_parser = new UAL::SXF::Parser();
$sxf_parser->read("$dir/$file" . ".sxf", "echo/$file" . ".sxf");
print "end                ", time, "\n";

# #############################################################
# Write a SXF file
# #############################################################

print "write the SXF file ", time, "\n";
$sxf_parser->write("out/$file" . ".sxf");
print "end                ", time, "\n";

# #############################################################
# Track partcile
# #############################################################

use RHIC::UI::Shell;

# Make the shell
my $shell = new RHIC::UI::Shell();

# Define DA Space
my $maxOrder = 2;
$shell->space($maxOrder);

# Select an accelerator  for operations
$shell->use($lattice);

# Define beam parameters

$shell->beam(energy => 450.0);

# Define initial particle coordinates
#$shell->start([1.0e-4, 0.0, 1.0e-4, 0.0, 0.0, 0.0]);

# Make and print a first turn track 
#$shell->firstturn("print"=> "./out/firstturn/$file" . ".sxf", observe => ""); 

# Define initial particle coordinates

my @bunch = ([1.0e-5, 0.0, 1.0e-5, 0.0, 0.0, 1.0e-5],
	     [1.0e-4, 0.0, 1.0e-4, 0.0, 0.0, 1.0e-4],
	     [1.0e-3, 0.0, 1.0e-3, 0.0, 0.0, 1.0e-3]);

$shell->start(@bunch);

# Track particles 
$shell->run("print" => ">./out/8/$file" . ".sxf", "turns" => 1, "step" => 1); 
