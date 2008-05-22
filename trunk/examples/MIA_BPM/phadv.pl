#!/usr/bin/perl

my $job_name   = "test_no_dxmp";

use File::Path;
use File::Basename;
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

$sxf_parser->read("./data/rhic_injection.sxf", "./out/" . $job_name . "/echo.sxf");

# ------------------------------------------------------
# Select and initialize a lattice for operations
# ------------------------------------------------------

print "Select and initialize a lattice", "\n";

# Select an accelerator for operations

$shell->use("lattice" => "blue");

# Write SMF into the FTPOT file

$shell->writeFTPOT("file" => "./out/" . $job_name . "/tpot");

# ------------------------------------------------------
# Define beam parameters
# ------------------------------------------------------

print "Define beam parameters", "\n";

$shell->setBeamAttributes("energy" => 250.0, "mass" => 0.93827231);

# ------------------------------------------------------
# Linear analysis
# ------------------------------------------------------

print "Linear analysis: ", "\n";

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

# $shell->twiss("elements" => "cplmon", "print" => "./out/" . $job_name . "/twiss"); 

print_phadv($shell, "./phadv.out", "cplmon", 0.0);

print_phadv($shell, "./quads.out", "^(q.*)\$", 0.0);

sub print_phadv
{
    my ($shell, $file, $regex, $de) = @_;


    my $lattice = $shell->{lattice};
    my $code = $shell->{code};
    my $beam = $shell->{beam};

    my $dir = dirname($file);

    # Closed orbit

    my $orbit = new Pac::Position(); 
    $orbit->de($de);
    $code->clorbit($orbit, $beam);


    # One-Turn Map 

    my $oneTurnMap = new Pac::TMap(6);

    $oneTurnMap->refOrbit($orbit);	
    $code->map($oneTurnMap, $beam, 1); 


    my $eigenMap = new Pac::TMap(6);
    
    open(TWISS, ">$file") || die "can't create file $file";


    my $twiss = new Pac::TwissData; 
    my $chrom = new Pac::ChromData;
    my ($i, $le, $suml, $bName) = (0, 0, 0, " ");

    my $pi2grad = 360.0*1./atan2(1,1)/8.;   

 
    $code->chrom($chrom, $beam, $orbit);
    $twiss = $chrom->twiss();
    $twiss->mu(0, 0.0);
    $twiss->mu(1, 0.0);

    my $counter = 0;
    my $prev_phadv = $twiss->mu(0);
    my ($mux, $muy) = (0.0, 0.0);
 
    for($i=0; $i < $lattice->size; $i++){
	$le = $lattice->element($i);    
    
	if($le->genName =~ $regex) {
	  $output = sprintf("%5d %5d %-10s %10.3e %10.3e\n", 
			    $counter++, $i, $le->genName(), 
			    ($twiss->mu(0) - $prev_phadv)*$pi2grad,
			     $twiss->beta(0));		
	  print TWISS $output;
	  $prev_phadv = $twiss->mu(0);

	}

	# track map
        my $sectorMap = new Pac::TMap(6); 
        $sectorMap->refOrbit($orbit);  

	$code->trackMap($sectorMap, $beam, $i, $i+1);
	$code->trackClorbit($orbit, $beam, $i, $i+1);

	# track twiss
	$code->trackTwiss($twiss, $sectorMap);

	# check mu
	if(($twiss->mu(0) - $mux) < 0.0) { $twiss->mu(0, $twiss->mu(0) + 1.0); }
	$mux = $twiss->mu(0);
	if(($twiss->mu(1) - $muy) < 0.0) { $twiss->mu(1, $twiss->mu(1) + 1.0); }
	$muy = $twiss->mu(1);

	# update suml
    }   
   

    close(TWISS);
}


print "End", "\n";

1;
