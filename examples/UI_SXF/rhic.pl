#!/usr/bin/perl

my $job_name   = "rhic";

use File::Path;
mkpath(["./out/" . $job_name], 1, 0755);

# ------------------------------------------------------
# Create the ALE::UI::Shell  instance 
# ------------------------------------------------------

use lib ("$ENV{UAL_EXTRA}/ALE/api");
use ALE::UI::Shell;

print "Create the ALE::UI::Shell instance", "\n";

my $shell = new ALE::UI::Shell("print" => "./out/" . $job_name . "/log");

# ------------------------------------------------------
# Define the space of Taylor maps
# ------------------------------------------------------

print "Define the space of Taylor maps", "\n";

$shell->setMapAttributes("order" => 5);

# ------------------------------------------------------
# Define design elements and beam lines
# ------------------------------------------------------

# Read the SXF file

print "read the SXF file   ", time, "\n";

use lib ("$ENV{UAL_SXF}/api");
use UAL::SXF::Parser;
my $sxf_parser = new UAL::SXF::Parser();

$sxf_parser->read("./data/blue-dAu-top-swn-no_sexts.sxf", "./out/" . $job_name . "/echo.sxf");

# ------------------------------------------------------
# Select and initialize a lattice for operations
# ------------------------------------------------------

print "Select and initialize a lattice", "\n";

# Select an accelerator for operations

$shell->use("lattice" => "RHIC");

# ------------------------------------------------------
# Add apertures
# ------------------------------------------------------

$shell->addAperture("elements" => "^dxmp", 
                    "aperture" => [1., 0.116, 0.079]);

# ------------------------------------------------------
# Write the lattice state to the SXF file
# ------------------------------------------------------

$sxf_parser->write("./out/" . $job_name . "/rhic.sxf");

# ------------------------------------------------------
# Define beam parameters
# ------------------------------------------------------

print "Define beam parameters", "\n";

$shell->setBeamAttributes("energy" => 250, "mass" => 0.93827231);

# ------------------------------------------------------
# Analysis
# ------------------------------------------------------

print "Linear analysis", "\n";

$shell->analysis("print" => "./out/" . $job_name . "/analysis"); 

print "Make the second-order map", "\n";

$shell->map("order" => 2, "print" => "./out/" . $job_name . "/map2"); 


print "End", "\n";

1;
