
#!/usr/bin/perl

my $job_name   = "ring-Oct-2003";
my $mad_file   = "./data/ring-Oct-2003.mad";
my $sxf_file   = "./data/ring-Oct-2003.sxf";

my $latticeName = "rng";
my $mass        = 0.93827231;
my $energy      = 3. + $mass; # GeV

my $tunex       = 22.4387;
my $tuney       = 20.8023; 

my $chromx      =  0.0;
my $chromy      =  0.0;

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

$shell->readMAD("file" => $mad_file);

# ------------------------------------------------------
# Select and initialize a lattice for operations
# ------------------------------------------------------

$shell->addSplit("elements" => "^q", "ir" => 2); 

# ------------------------------------------------------
# Select and initialize a lattice for operations
# ------------------------------------------------------

print "Select and initialize a lattice (", __LINE__, ")\n";

# Select an accelerator for operations

$shell->use("lattice" => $latticeName);

# Write SMF into the FTPOT file

$shell->writeFTPOT("file" => "./out/" . $job_name . "/tpot");

# ------------------------------------------------------
# Define beam parameters
# ------------------------------------------------------

print "Define beam parameters (", __LINE__, ")\n";

$shell->setBeamAttributes("energy" => $energy, "mass" => $mass);

# ------------------------------------------------------
# Linear analysis
# ------------------------------------------------------

print "Linear analysis (", __LINE__, ")\n";

# Make general analysis
print " analysis\n";
$shell->analysis("print" => "./out/" . $job_name . "/analysis", "dp/p" => 0.0);
$shell->analysis("print" => "./out/" . $job_name . "/analysis_de", "dp/p" => 0.0101397);

# Make linear matrix
print " matrix\n";
# $shell->map("order" => 1, "print" => "./out/" . $job_name . "/map1", "de" => 0.0); 
# $shell->map("order" => 1, "print" => "./out/" . $job_name . "/map1_de", "de" => 0.0101397);

# Calculate survey
print " survey\n";

$shell->survey("elements" => "", "print" => "./out/" . $job_name . "/survey"); 

# Calculate twiss
print " twiss\n";
$shell->twiss("elements" => "", "print" => "./out/" . $job_name . "/twiss"); 

# --------------------------------------------------------------------
print "\nAdjust tune ...\n";
# --------------------------------------------------------------------

#
#$shell->tunethin("bf" => "^qf\$", "bd" => "^qd\$",
#	         "mux" => $tunex, "muy" => $tuney);
#$shell->analysis("print" => "./out/" . $job_name . "/tune.analysis" );

#
# --------------------------------------------------------------------
print "\nAdjust chromaticity ...\n";
# --------------------------------------------------------------------
$shell->chromfit("bf" => "^sxf\$", "bd" => "^sxd\$",
                  "chromx" => $chromx, "chromy" => $chromy);
$shell->analysis("print" => "./out/" . $job_name . "/chrom.analysis" );

# ------------------------------------------------------
# Write a SXF file
# ------------------------------------------------------

print "Store the accelerator data into the SXF format (", __LINE__, ")\n";

use lib ("$ENV{UAL_SXF}/api");
use UAL::SXF::Parser;
my $sxf_parser = new UAL::SXF::Parser();
$sxf_parser->write($sxf_file);

# ------------------------------------------------------
# Track bunch of particles 
# ------------------------------------------------------

print "Track bunch of particles (", __LINE__, ")\n";

my ($i, $size, $irs, $ias) = (0, 90, 10, 9);
my $bunch = new ALE::UI::Bunch($size);

my ($ir, $ia, $rem, $phi, $exi, $eyi, $xi, $yi);

my $ex = 54.0e-6;
my $ey = 54.0e-6;

my $betax = 1.3915e+01;
my $betay = 1.3179e+01;

my $alphax =  1.5794e+00;
my $alphay = -1.3843e+00;

my $gammax = (1. + $alphax*$alphax)/$betax;
my $gammay = (1. + $alphay*$alphay)/$betay;

my $de = 0.007;

my $ip = 0;
for($ir = 0; $ir < $irs; $ir++){
  for($ia = 0; $ia < $ias; $ia++){

      $rem  = ($ir+1.0)/$irs;
      $phi  = ($ia+1.0)*(3.14159/2.)/($ias + 1);

      $exi  = $rem*cos($phi);
      $eyi  = $rem*sin($phi);

      $xi   = sqrt($ex*$exi/$gammax);
      $yi   = sqrt($ey*$eyi/$gammay);

      # index, 6 coordinates
      $bunch->setPosition($ip++, $xi, 0.0, $yi, 0.0, 0.0, $bunch->_de2dp($de));
  }
}

$bunch->setBeamAttributes($energy, $mass);

# for($i =0; $i < $size; $i++){
#   $bunch->setPosition($i, 1.e-3, 0.0, 1.e-3, 0.0, 0.0, 0.0);
# }
# $bunch->setPosition(0, 0.00620549, -0.00188178, -0.00694747, 0.000277732, 0.0, 0.0101397);

$shell->run("turns" => 10, "step" => 1, 
	    "print" => "./out/" . $job_name . "/fort.8", "bunch" => $bunch);

open(BUNCH_OUT, ">./out/" . $job_name . "/bunch_out_new") || die "can't create file(bunch_out_new)";

my @p;
for($i =0; $i < $size; $i++){
  if($bunch->{"bunch"}->flag($i) > 1) { print BUNCH_OUT "$i is lost particle \n"; }
    @p = $bunch->getPosition($i);
    $output= sprintf
    ("i=%5d x=%14.8e px=%14.8e y=%14.8e py=%14.8e ct=%14.8e dp/p=%14.8e \n",
     $i,    $p[0],   $p[1],    $p[2],    $p[3],    $p[4],     $bunch->_dp2de($p[5])); 
     print BUNCH_OUT $output;
}

# $shell->firstturn("observe" => "", "particle" => [0.00620549, -0.00188178, -0.00694747, 0.000277732, 0.0, 0.0101397]);
print "End (", __LINE__, ")\n";

