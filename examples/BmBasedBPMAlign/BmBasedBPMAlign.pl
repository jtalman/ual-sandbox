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

$shell->setMapAttributes("order" => 6);

# ------------------------------------------------------
# Define design elements and beam lines
# ------------------------------------------------------

# Read MAD input file

print "Read MAD input file", "\n";

$shell->readMAD("file" => "./data/BmBasedBPMAlign.mad");

# Split generic elements into thin multipoles

$shell->addSplit("elements" => "^(q[df]h|q[fd][lmc]h|qfbh)\$", 
                 "ir" => 1); 
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
# Define beam parameters
# ------------------------------------------------------

print "Define beam parameters", "\n";

$shell->setBeamAttributes("energy" => 1.93827231, "mass" => 0.93827231);

# ------------------------------------------------------
# Linear analysis
# ------------------------------------------------------

print "Linear analysis", "\n";

# Make general analysis

my $dp=0;
# for($dp = -0.02; $dp <= 0.02; $dp += 0.005){
# for($dp = 0.0; $dp <= 0.0; $dp += 0.005){
#   $shell->analysis("print" => "./out/" . $job_name . "/analysis" . "." . $dp, "dp# /p" => $dp); 
# }

# Make linear matrix

$shell->map("order" => 1, "print" => "./out/" . $job_name . "/map1"); 

# ------------------------------------------------------
# Add systematic errors 
# ------------------------------------------------------

print "Add systematic errors", "\n";

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

print "Add random errors", "\n";

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
# Investigation of beam-based BPM alignment 
# ------------------------------------------------------

my $iseed = 973431;
my $rgenerator = new ALE::UI::RandomGenerator($iseed);

print "Investigate beam-based BPM alignment", "\n";

# At this point various errors have been applied, but none affect
# the closed orbit, so the orbit is known to be smooth in both planes.

# Model random BPM misalignment by applying random misalignment
# of the 16 QFH half-quadrupoles. They make up a family of quads
# whose strengths can only be varied together by equal amounts
# because their trim windings are ganged together.

print "Misalign all QFH half-quads, and print net deflections", "\n";

my ($rMisalignIndices,$rdelx,$rdely,$rdeltheta) = 
  $shell->addMisalignment("elements" => "^qfh\$", "dx" => 0.01, "dy" => 0.01);

my $numMisaligns = $#{$rMisalignIndices};
my $numkicks = ($numMisaligns + 1)/2;

my ($rQuadIndices,$rklvaluebef) = $shell->getErectMagnetStrengths("elements" => "^qfh\$", "multindex" => 1);

$shell->analysis("print" => "./out/" . $job_name . "/analysis" . "." . $dp, "dp/p" => $dp);

# The following resteers the beam and prints out the strengths 

print  "\n", "Re-steer beam and print required deflections";

$shell->hsteer("adjusters" => "^kickh", "detectors" => "^bpmh");
$shell->vsteer("adjusters" => "^kickv", "detectors" => "^bpmv");
$shell->hsteer("adjusters" => "^kickh", "detectors" => "^bpmh");
$shell->vsteer("adjusters" => "^kickv", "detectors" => "^bpmv");

my ($rErectIndices,$rkickhs) = $shell->getErectMagnetStrengths("elements" => "^kickh", "multindex" => 0);
my ($rSkewIndices,$rkickvs) = $shell->getSkewMagnetStrengths("elements" => "^kickv", "multindex" => 0);

print ("q_num el_index  kl1_bef   del_x     del_y     defl_x    kickh     defl_y    kickv \n");
for($ik=0; $ik < $numkicks; $ik++){
   $i = 2*$ik;
   $defl_x[$ik] = -(${$rdelx}[$i] + ${$rdelx}[$i+1])*${$rklvaluebef}[$i];
   $defl_y[$ik] =  (${$rdely}[$i] + ${$rdely}[$i+1])*${$rklvaluebef}[$i];
   printf ("%4.0f %8.0f %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f\n", 
	   $i, 
	   ${$rMisalignIndices}[$i], 
           ${$rklvaluebef}[$i], 
           ${$rdelx}[$i], 
	   ${$rdely}[$i], 
	   $defl_x[$ik], 
	   ${$rkickhs}[$ik], 
	   $defl_y[$ik], 
	   ${$rkickvs}[$ik] );
   printf ("%4.0f %8.0f %9.5f %9.5f %9.5f\n", 
	   $i+1, 
	   ${$rMisalignIndices}[$i+1],
           ${$rklvaluebef}[$i+1], 
           ${$rdelx}[$i+1], 
	   ${$rdely}[$i+1] );
}

# The fact that the strengths determined by hsteer and vsteer agree well 
# with the randomly applied deflections suggests that the beam-based
# re-alignment will work. But the actual diagnostic procedure has
# not yet been modeled.

# Apply standard trim current to all "QFH" quadrupoles
# 100 units of b1 give 1 percent

my $f = 0.01;
my $dqInUnits = 10000.0*$f;

print "\nApply standard trim current to all QFH quadrupoles", "\n";
$shell->addFieldError("elements" => "^qfh\$",  "R" => 0.1, "b" => [0.0, $dqInUnits]);
my ($rQuadIndices,$rklvalueaft) = $shell->getErectMagnetStrengths("elements" => "^qfh\$", "multindex" => 1);

print ("q_num el_index  kl1_aft \n");
for($ik=0; $ik < $numkicks; $ik++){
   $i = 2*$ik;
   printf ("%4.0f %8.0f %9.5f\n", $i, ${$rMisalignIndices}[$i], ${$rklvalueaft}[$i] );
   printf ("%4.0f %8.0f %9.5f\n", $i+1, ${$rMisalignIndices}[$i+1],${$rklvalueaft}[$i+1] );
};

$shell->analysis("print" => "./out/" . $job_name . "/analysis" . "." . $dp, "dp/p" => $dp);

print  "\n", "Re-steer beam and print required deflections";

$shell->hsteer("adjusters" => "^kickh", "detectors" => "^bpmh");
$shell->vsteer("adjusters" => "^kickv", "detectors" => "^bpmv");
$shell->hsteer("adjusters" => "^kickh", "detectors" => "^bpmh");
$shell->vsteer("adjusters" => "^kickv", "detectors" => "^bpmv");

my ($rErectIndices,$rkickhafts) = $shell->getErectMagnetStrengths("elements" => "^kickh", "multindex" => 0);
my ($rSkewIndices, $rkickvafts) = $shell->getSkewMagnetStrengths("elements" => "^kickv", "multindex" => 0);

print ("q_num el_index  del_kick_x  -defl_x*f |  del_kick_y   defl_y*f \n");
for($ik=0; $ik < $numkicks; $ik++){
   printf ("%4.0f %8.0f %11.7f %11.7f |%11.7f %11.7f\n", 
	   $ik, ${$rMisalignIndices}[2*$ik], 
           ${$rkickhafts}[$ik]-${$rkickhs}[$ik], 
           -$defl_x[$ik]*$f, 
           ${$rkickvafts}[$ik]-${$rkickvs}[$ik], 
           $defl_y[$ik]*$f );
}

# ------------------------------------------------------
# Track bunch of particles 
# ------------------------------------------------------

print "Track bunch of particles", "\n";

my ($i, $size) = (0, 10);

my $bunch = new ALE::UI::Bunch($size);

$bunch->setBeamAttributes(1.93827231, 0.93827231);

for($i =0; $i < $size; $i++){
    $bunch->setPosition($i, 1.e-2*$i, 0.0, 1.e-2*$i, 0.0, 0.0, 1.e-3*$i);
}

$shell->run("turns" => 100, "step" => 10, "print" => "./out/" . $job_name . "/fort.8",
  	    "bunch" => $bunch);

open(BUNCH_OUT, ">./out/" . $job_name . "/bunch_out_new") || die "can't create file(bunch_out_mpi)";

my @p;
for($i =0; $i < $size; $i++){
    @p = $bunch->getPosition($i);
    $output= sprintf("i=%5d x=%14.8e px=%14.8e y=%14.8e py=%14.8e ct=%14.8e dp/p=%14.8e \n",
                         $i,$p[0],      $p[1],    $p[2],    $p[3],    $p[4],     $p[5]); 
     print BUNCH_OUT $output;
}

print "End", "\n";

1;
