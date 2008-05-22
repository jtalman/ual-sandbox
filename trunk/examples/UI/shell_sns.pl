#!/usr/bin/perl

my $job_name   = "test";

use File::Path;
mkpath(["./out/" . $job_name], 1, 0755);

# ------------------------------------------------------
# Create the ALE::UI::Shell  instance 
# ------------------------------------------------------

use lib ("$ENV{UAL_EXTRA}/ALE/api");
use ALE::UI::Shell;

print "Create the ALE::UI::Shell instance (", __LINE__,  ")\n";

my $shell = new ALE::UI::Shell("print" => "./out/" . $job_name . "/log");

# ------------------------------------------------------
# Define the space of Taylor maps
# ------------------------------------------------------

print "Define the space of Taylor maps (", __LINE__, ")\n";

$shell->setMapAttributes("order" => 5);

# ------------------------------------------------------
# Define design elements and beam lines
# ------------------------------------------------------

# Read MAD input file

print "Read MAD input file (", __LINE__, ")\n";

$shell->readMAD("file" => "./data/ff_sext_latnat.mad");

# Split generic elements into thin multipoles

$shell->addSplit("elements" => "^(q[df]h|q[fd][lmc]h|qfbh)\$", 
                 "ir" => 2); 
$shell->addSplit("elements" => "^(bnd)\$", 
                 "ir" => 2); 

# Define aperture parameters: shape, xsize, and ysize.

print "Define aperture parameters (", __LINE__, ")\n";

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

print "Select and initialize a lattice (", __LINE__, ")\n";

# Select an accelerator for operations

$shell->use("lattice" => "ring");

# Write SMF into the FTPOT file

$shell->writeFTPOT("file" => "./out/" . $job_name . "/tpot");

# ------------------------------------------------------
# Define beam parameters
# ------------------------------------------------------

print "Define beam parameters (", __LINE__, ")\n";

$shell->setBeamAttributes("energy" => 1.93827231, "mass" => 0.93827231);

# ------------------------------------------------------
# Linear analysis
# ------------------------------------------------------

print "Linear analysis (", __LINE__, ")\n";

# Make general analysis

my $dp;
for($dp = -0.02; $dp <= 0.02; $dp += 0.005){
  $shell->analysis("print" => "./out/" . $job_name . "/analysis" . "." . $dp, 
  "dp/p" => $dp); 
}

# Make linear matrix

$shell->map("order" => 1, "print" => "./out/" . $job_name . "/map1"); 


# ------------------------------------------------------
# Add systematic errors 
# ------------------------------------------------------

print "Add systematic errors (", __LINE__, ")\n";

# dipoles

$shell->addFieldError("elements" => "^(bnd)\$",   "R" => 0.13,
                      "b" => [0.0, 0.1, 51.0, 0.5, -26.0, 0.2, 0.0, 0.0, 0.0, 0.0]);


# regular arc and matching quads

$shell->addFieldError("elements" => "^(q[df]h|qdmh)\$",  "R" => 0.1,
                      "b" => [0.0, 0.0, 0.4, 0.1, 0.7, -12.10, 0.0, 0.0, 0.0, 0.0]);

# large arc quad and doublet quads

$shell->addFieldError("elements" => "^(q[fd][lc]h|qfbh)\$",  "R" => 0.12,
                      "b" => [0.0, 0.0, 0.4, 0.1, 0.7, -12.10, 0.0, 0.0, 0.0, 0.0]);

# ------------------------------------------------------
# Add random errors 
# ------------------------------------------------------

print "Add random errors (", __LINE__, ")\n";

my $iseed = 973431;
my $rgenerator = new ALE::UI::RandomGenerator($iseed);

# Quadrupole random field errors

my $qSigB  = [0.0, 0.0, -2.46, -0.76, -0.63, 0.00, 0.02, -0.63,  0.17, 0.00];
my $qSiqA = [0.0, 0.0, -2.50, -2.00,  1.29, 1.45, 0.25,  0.31, -0.11, 1.04];

# Quads

$shell->addFieldError("elements" => "^qdmh",  "R" => 0.1,
                      "b" => $qSigB, "a" => $qSiqA, "engine" => $rgenerator);

$shell->addFieldError("elements" => "^qdh",  "R" => 0.1,
                      "b" => $qSigB, "a" => $qSiqA, "engine" => $rgenerator);

$shell->addFieldError("elements" => "^qfh",  "R" => 0.1,
                      "b" => $qSigB, "a" => $qSiqA, "engine" => $rgenerator);

$shell->addFieldError("elements" => "^qfbh",  "R" => 0.13,
                      "b" => $qSigB, "a" => $qSiqA, "engine" => $rgenerator);

$shell->addFieldError("elements" => "^qflh",  "R" => 0.13,
                       "b" => $qSigB, "a" => $qSiqA, "engine" => $rgenerator);

$shell->addFieldError("elements" => "^qdch", "R" => 0.13,
                      "b" => $qSigB, "a" => $qSiqA, "engine" => $rgenerator);

# ------------------------------------------------------
# Track bunch of particles 
# ------------------------------------------------------

print "Track bunch of particles (", __LINE__, ")\n";

my ($i, $size) = (0, 10);

my $bunch = new ALE::UI::Bunch($size);

$bunch->setBeamAttributes(1.93827231, 0.93827231);

for($i =0; $i < $size; $i++){
    $bunch->setPosition($i, 1.e-2*$i, 0.0, 1.e-2*$i, 0.0, 0.0, 1.e-3*$i);
}

$shell->run("turns" => 100, "step" => 10, 
	    "print" => "./out/" . $job_name . "/fort.8", "bunch" => $bunch);

open(BUNCH_OUT, ">./out/" . $job_name . "/bunch_out_new") 
  || die "can't create file(bunch_out_mpi)";

my @p;
for($i =0; $i < $size; $i++){
    @p = $bunch->getPosition($i);
    $output= sprintf
    ("i=%5d x=%14.8e px=%14.8e y=%14.8e py=%14.8e ct=%14.8e dp/p=%14.8e \n",
     $i,$p[0],      $p[1],    $p[2],    $p[3],    $p[4],     $p[5]); 
     print BUNCH_OUT $output;
}

print "End (", __LINE__, ")\n";

1;
