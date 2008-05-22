
#!/usr/bin/perl

my $job_name   = "mad2sxf";

use File::Path;
mkpath(["./out/" . $job_name], 1, 0755);

# ------------------------------------------------------
# Create the ALE::UI::Shell  instance 
# ------------------------------------------------------

use lib ("./api");
use Gt::Shell;

print "Create the Gt::Shell instance (", __LINE__,  ")\n";

my $shell = new Gt::Shell("print" => "./out/" . $job_name . "/log");

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

$shell->readMAD("file" => "./data/MAD8_lat.asc");

# strength10
# 

my $G4M   =  9.0658687891e-02; # noc
my $G56M  =  9.0346825648e-02; # noc
my $GD;
my $GF;
my $GDA   =  7.9348470640e-02; # noc
my $GFA   =  7.8179982140e-02; # noc
my $GFB   =  7.7658182372e-02; # noc
my $G7M   =  9.0065614398e-02; # noc
my $G6I   =  2.0785187925e-02; # noc
my $G5I   =  2.3111462439e-02; # noc
my $G4I   = -2.3121582536e-02; # noc
my $G6O   =  2.0096798328e-02; # noc
my $G5O   =  2.3111462439e-02; # noc
my $G4O   = -2.3121582536e-02; # noc
my $G3M   =  5.4639728243e-02; # noc
my $G2M   =  5.5383483724e-02; # noc
my $G1M   =  5.5738226820e-02; # noc


my @noc_params = ($G4M, $G56M, $GDA, $GFA, $GFB, 
		  $G7M, $G6O, $G5O, $G4O, $G6I, 
		  $G5I, $G4I, $G3M, $G2M, $G1M);

$shell->setNoc( 6, @noc_params);
$shell->setNoc( 8, @noc_params);
$shell->setNoc(10, @noc_params);
$shell->setNoc(12, @noc_params);
$shell->setNoc( 2, @noc_params);
$shell->setNoc( 4, @noc_params);

# set arc quads

my $KD = -8.414758E-02;
my $KF =  8.148073E-02;

$shell->setQuad("qf",  $KF);
$shell->setQuad("qd",  $KD);

# Setting sextupoles

my $SD0 = -3.059554E-01;
my $SF0 =  1.638809E-01;

$shell->setchrom($SF0/2., $SD0/2.);


# ------------------------------------------------------
# Select and initialize a lattice for operations
# ------------------------------------------------------

# $shell->addSplit("elements" => "^d", "ir" => 1); 

# ------------------------------------------------------
# Select and initialize a lattice for operations
# ------------------------------------------------------

print "Select and initialize a lattice (", __LINE__, ")\n";

# Select an accelerator for operations

$shell->use("lattice" => "yellow");

# Write SMF into the FTPOT file

$shell->writeFTPOT("file" => "./out/" . $job_name . "/tpot");

# ------------------------------------------------------
# Define beam parameters
# ------------------------------------------------------

print "Define beam parameters (", __LINE__, ")\n";

$shell->setBeamAttributes("energy" => 0.93827*268.2, "mass" => 0.93827);

# ------------------------------------------------------
# Linear analysis
# ------------------------------------------------------

print "Linear analysis (", __LINE__, ")\n";

# Make general analysis
print " analysis\n";
$shell->analysis("print" => "./out/" . $job_name . "/analysis");

# Make linear matrix
print " matrix\n";
$shell->map("order" => 1, "print" => "./out/" . $job_name . "/map1"); 

# Calculate survey
print " survey\n";

$shell->survey("elements" => "", "print" => "./out/" . $job_name . "/survey"); 

# Calculate twiss
print " twiss\n";
$shell->twiss("elements" => "", "print" => "./out/" . $job_name . "/twiss"); 

# ------------------------------------------------------
# Write a SXF file
# ------------------------------------------------------

print "Store the accelerator data into the SXF format (", __LINE__, ")\n";

use lib ("$ENV{UAL_SXF}/api");
use UAL::SXF::Parser;
my $sxf_parser = new UAL::SXF::Parser();
$sxf_parser->write("./data/rhic.sxf");

print "End (", __LINE__, ")\n";

