#!/usr/bin/perl

my $job_name   = "pl";

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

# Read MAD input file

print "Read MAD input file", "\n";

$shell->readMAD("file" => "./data/ff_sext_latnat.mad");

# Split generic elements into thin multipoles

$shell->addSplit("elements" => "^(q[df]h|q[fd][lmc]h|qfbh)\$", 
                 "ir" => 2); 
$shell->addSplit("elements" => "^(bnd)\$", 
                 "ir" => 2); 

# Define aperture parameters: shape, xsize, and ysize.

print "Define aperture parameters", "\n";

# half dipoles

$shell->addAperture("elements" => "^(bnd)\$", 
                    "aperture" => [1., 0.116, 0.079]);

# half quads

$shell->addAperture("elements" => "^(q[fd]h|qdmh)\$", 
                    "aperture" => [1., 0.105, 0.105]);


$shell->addAperture("elements" => "^(qfbh)\$", 
                    "aperture" => [1., 0.13, 0.13]);

$shell->addAperture("elements" => "^(q[fd][lc]h)\$", 
                    "aperture" => [1., 0.15, 0.15]);


$shell->addAperture("elements" => "^(s[13][fd])\$", 
                    "aperture" => [1., 0.105, 0.105]);


$shell->addAperture("elements" => "^(s[24][fd])\$", 
                    "aperture" => [1., 0.13, 0.13]);


# ------------------------------------------------------
# Select and initialize a lattice for operations
# ------------------------------------------------------

print "Select and initialize a lattice", "\n";

# Select an accelerator for operations

$shell->use("lattice" => "ring");

# Write SMF into the FTPOT file

$shell->writeFTPOT("file" => "./out/" . $job_name . "/tpot");

# ------------------------------------------------------
# Write the lattice state to the SXF file
# ------------------------------------------------------

use lib ("$ENV{UAL_EXTRA}/ADXF/api");
use UAL::ADXF::Parser;

my $adxf_parser = new UAL::ADXF::Parser();


$adxf_parser->write("./out/" . $job_name . "/ff_sext_latnat.adxf");

# ------------------------------------------------------
# Define beam parameters
# ------------------------------------------------------

print "Define beam parameters", "\n";

$shell->setBeamAttributes("energy" => 1.93827231, "mass" => 0.93827231);

# ------------------------------------------------------
# Linear analysis
# ------------------------------------------------------

print "Linear analysis: ", "\n";

# Make general analysis
print " analysis\n";

my $dp;
for($dp = -0.02; $dp <= 0.02; $dp += 0.005){
  $shell->analysis("print" => "./out/" . $job_name . "/analysis" . "." . $dp, "dp/p" => $dp); 
}

# Make linear matrix
print " matrix\n";

$shell->map("order" => 1, "print" => "./out/" . $job_name . "/map1"); 

# Calculate survey
print " survey\n";

$shell->survey("elements" => "", "print" => "./out/" . $job_name . "/survey"); 

# Calculate twiss
print " twiss\n";

$shell->twiss("elements" => "", "print" => "./out/" . $job_name . "/twiss"); 

print "End", "\n";

1;
