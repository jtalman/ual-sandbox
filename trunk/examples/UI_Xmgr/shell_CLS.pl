#!/usr/bin/perl

my $job_name   = "test";

use File::Path;
mkpath(["./out/" . $job_name], 1, 0755);

# ------------------------------------------------------
# Create the UALUSR::Shell  instance 
# ------------------------------------------------------

use lib ("./api");
use UALUSR::Shell;

print "Create the UALUSR::Shell instance", "\n";

my $shell = new UALUSR::Shell("print" => "./out/" . $job_name . "/log");

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

$shell->readMAD("file" => "./data/CLS.mad");

# Split generic elements into thin multipoles

$shell->addSplit("elements" => "^(quad1|quad2|quad3)\$", 
                 "ir" => 2); 
$shell->addSplit("elements" => "^(bend)\$", 
                 "ir" => 2); 

# Define aperture parameters: shape, xsize, and ysize.

print "Define aperture parameters", "\n";

$shell->addAperture("elements" => "^(hc)\$", 
                    "aperture" => [1., 0.035, 0.035]);

$shell->addAperture("elements" => "^(q1|q2|q3)\$", 
                    "aperture" => [1., 0.035, 0.035]);

$shell->addAperture("elements" => "^(s1|s2)\$", 
                    "aperture" => [1., 0.035, 0.035]);

# ------------------------------------------------------
# Select and initialize a lattice for operations
# ------------------------------------------------------

print "Select and initialize a lattice", "\n";

# Select an accelerator for operations

$shell->use("lattice" => "cell");

# Write SMF into the FTPOT file

$shell->writeFTPOT("file" => "./out/" . $job_name . "/tpot");

# ------------------------------------------------------
# Define beam parameters
# ------------------------------------------------------

print "Define beam parameters", "\n";

$shell->setBeamAttributes("energy" => 2.9, "mass" => 0.51099906);

$shell->survey("elements" => "", "print" =>"./out/" . $job_name . "/survey");

# $shell->tunethin("bf" => "^qt\$", "bd" => "^m1hc\$",
#                  "mux" => 10.22, "muy" => 3.26);

$shell->twiss("elements" => "^(quad[1-3]|quadin|quadout|bend|bendin|bendout|bendedge|sext|cellend|testpoint[1-2])", "print" =>"./out/" . $job_name . "/twiss");

# ------------------------------------------------------
# Linear analysis
# ------------------------------------------------------

print "Linear analysis", "\n";

# Make general analysis

$shell->analysis("print" => "./out/" . $job_name . "/analysis");

# Make linear matrix

$shell->map("order" => 1, "print" => "./out/" . $job_name . "/map1"); 

# ------------------------------------------------------
# Track bunch of particles 
# ------------------------------------------------------

print "Track bunch of particles", "\n";

my ($i, $size) = (0, 1);

my $bunch = new ALE::UI::Bunch($size);

$bunch->setBeamAttributes(1.93827231, 0.93827231);

for($i =0; $i < $size; $i++){
    $bunch->setPosition($i, 1.e-2*($i+1), 0.0, 1.e-2*($i+1), 0.0, 0.0, 1.e-3*($i+1));
}

$shell->run("turns" => 256, "step" => 1, "print" => "./out/" . $job_name . "/fort.8",
  	    "bunch" => $bunch);

# ------------------------------------------------------------------------
# Plot turn-by-turn data
# ------------------------------------------------------------------------
system 'xmgrace -noask -b xyzfft.com &';



print "End", "\n";

1;
