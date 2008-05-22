#!/usr/bin/perl

use lib ("$ENV{UAL_MPI_PERL}/api");
use Short_MPI;

# -------------------------------------------------------------------
#Create the MPI environment 
#Define the total number of MPI processes available - $numprocs 
#Define the rank of the calling process in group    - $myid
# -------------------------------------------------------------------

my @mpi_argv = ($0, @ARGV);
my $mpi_argc = $#mpi_argv + 1;
Short_MPI::MPI_Init($mpi_argc, \@mpi_argv);

my $process_mun;
Short_MPI::MPI_Comm_size($Short_MPI::MPI_COMM_WORLD, \$process_mun);

my $process_id;
Short_MPI::MPI_Comm_rank($Short_MPI::MPI_COMM_WORLD, \$process_id);

printf(STDERR "Process starts on the node %d (host = %s)\n", $process_id, $ENV{HOST});

Short_MPI::MPI_Barrier($Short_MPI::MPI_COMM_WORLD);
my $time_total_start = Short_MPI::MPI_Wtime(); 

# ------------------------------------------------------
# start of the shell_sns script
# ------------------------------------------------------

my $job_name   = "test";

use File::Path;

if($process_id == 0) {
  mkpath(["./out/" . $job_name], 1, 0755);
}

# ------------------------------------------------------
# Create the ALE::UI::Shell  instance 
# ------------------------------------------------------

use lib ("$ENV{UAL_EXTRA}/ALE/api");
use ALE::UI::Shell;

my $shell = new ALE::UI::Shell("print" => "./out/" . $job_name . "/log_" . $process_id);

# ------------------------------------------------------
# Define the space of Taylor maps
# ------------------------------------------------------

$shell->setMapAttributes("order" => 6);

# ------------------------------------------------------
# Define design elements and beam lines
# ------------------------------------------------------

# Read MAD input file

$shell->readMAD("file" => "./data/ff_sext_latnat.mad", "id" => $process_id);

# Split generic elements into thin multipoles

$shell->addSplit("elements" => "^(q[df]h|q[fd][lmc]h|qfbh)\$", 
                 "ir" => 2); 
$shell->addSplit("elements" => "^(bl|br)\$", 
                 "ir" => 2); 

# Define aperture parameters: shape, xsize, and ysize.

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

# Select an accelerator for operations

$shell->use("lattice" => "ring");

# Write SMF into the FTPOT file

$shell->writeFTPOT("file" => "./out/" . $job_name . "/tpot_" . $process_id );

# ------------------------------------------------------
# Define beam parameters
# ------------------------------------------------------

$shell->setBeamAttributes("energy" => 1.93827231, "mass" => 0.93827231);

# ------------------------------------------------------
# Linear analysis
# ------------------------------------------------------

# Make general analysis

my $dp;
for($dp = -0.02; $dp <= 0.02; $dp += 0.005){
  $shell->analysis("print" => "./out/" . $job_name . "/analysis" . "." . $dp . "_"  . $process_id, 
		   "dp/p" => $dp); 
}

# Make linear matrix

$shell->map("order" => 1, "print" => "./out/" . $job_name . "/map1_"  . $process_id ); 


# ------------------------------------------------------
# Add systematic errors 
# ------------------------------------------------------

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

my ($i, $size) = (0, 2);

my $bunch = new ALE::UI::Bunch($size);

$bunch->setBeamAttributes(1.93827231, 0.93827231);

for($i =0; $i < $size; $i++){
    $bunch->setPosition($i, 1.e-2*$i, 0.0, 1.e-2*$i, 0.0, 0.0, 1.e-3*$i);
}

$shell->run("turns" => 100, "step" => 10, "print" => "./out/" . $job_name . "/fort.8_" . $process_id,
  	    "bunch" => $bunch);


open(BUNCH_OUT, ">./out/" . $job_name . "/bunch_out_new_" .  $process_id) || 
  die "can't create file(bunch_out_new)";

my @p;
for($i =0; $i < $size; $i++){
    @p = $bunch->getPosition($i);
    $output= sprintf("i=%5d x=%14.8e px=%14.8e y=%14.8e py=%14.8e ct=%14.8e dp/p=%14.8e \n",
                         $i,$p[0],      $p[1],    $p[2],    $p[3],    $p[4],     $p[5]); 
     print BUNCH_OUT $output;
}

close(BUNCH_OUT);

# ------------------------------------------------------
# end of the shell_sns script
# ------------------------------------------------------

my $time_total_stop = Short_MPI::MPI_Wtime(); 

my $time_total = $time_total_stop - $time_total_start;
print "Node ", $process_id ,  ": total time =", $time_total, " \n";

# ------------------------------------------------------
# Finalize MPI
# ------------------------------------------------------

printf(STDERR "Process stops on the node = %d\n", $process_id);

Short_MPI::MPI_Finalize();


1;
