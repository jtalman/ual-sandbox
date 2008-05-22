#!/usr/bin/perl

# -------------------------------------------------------------------
# Define location of the MPI environment 
# -------------------------------------------------------------------

use lib ("$ENV{UAL_MPI_PERL}/api");

use Short_MPI;

#---------------------------------------------------------------------

use lib ("$ENV{UAL_PAC}/api", "$ENV{UAL_TEAPOT}/api", "$ENV{UAL_ZLIB}/api");
 
use Pac;  
use Teapot; 
use Zlib::Tps; 

# -------------------------------------------------------------------
#Create the MPI environment 
#Define the total number of MPI processes available - $numprocs 
#Define the rank of the calling process in group    - $myid
# -------------------------------------------------------------------

MPI_Init($0, @ARGV);
$numprocs = MPI_Comm_size();
$myid =     MPI_Comm_rank();

printf(STDERR "Process %d\n", $myid);

#--------------------------------------------------------------------



# ACCELERATOR DESCRIPTION

# Permanent part
require './local/ring.pl';

# Variable part
require './local/migrator.pl';

# ACTIONS

# Here you can call your favorite C/C++ accelerator 
# libraries ( Tracking, ZLIB(DA), Analysis, etc. )

$teapot = new Teapot::Main;
$teapot->use($ring);
$teapot->makethin();

# Make survey

require '../util/survey_mpi.pl';

# Track particles

require './local/tracking_mpi.pl';

# ------------------------------------------------------
# Finalize MPI
# ------------------------------------------------------

MPI_Finalize();

1;

