#!/usr/bin/perl

use lib ("$ENV{UAL_MPI_PERL}/api");
use Short_MPI;

# -------------------------------------------------------------------
#Create the MPI environment 
#Define the total number of MPI processes available - $numprocs 
#Define the rank of the calling process in group    - $myid
# -------------------------------------------------------------------

my $status_mpi;
Short_MPI::MPI_Initialized(\$status_mpi);
print "Status MPI (before MPI_Init) = ",$status_mpi,"\n";

my @mpi_argv = ($0, @ARGV); 
my $mpi_argc = $#mpi_argv + 1;
Short_MPI::MPI_Init($mpi_argc, \@mpi_argv);

my $numprocs;
Short_MPI::MPI_Comm_size($Short_MPI::MPI_COMM_WORLD, \$numprocs);

my $myid;
Short_MPI::MPI_Comm_rank($Short_MPI::MPI_COMM_WORLD, \$myid);

printf(STDERR "Process starts on the node %d, (host = %s)\n", $myid, $ENV{HOST});

Short_MPI::MPI_Initialized(\$status_mpi);

if($myid == 0) {
  print "Status MPI (after MPI_Init) = ", $status_mpi, 
        ", number of nodes = ", $numprocs, " \n";
}

#####################################################################
print "\nReplace these print statements by any Perl instructions\n";
print "For example cut and paste the entire contents of \n";
print "   $ENV{UAL}/examples/UI/shell_sns.pl \n";
print "(see $ENV{UAL}/examples/UI_MPI/shell_sns_mpi.pl)\n\n";
#####################################################################

Short_MPI::MPI_Barrier($Short_MPI::MPI_COMM_WORLD);
my $timeStop = Short_MPI::MPI_Wtime();
my $time_proc = $timeStop - $timeStart;
printf(STDERR "Node %d : time = %e \n", $myid, $time_proc );

# -------------------------------------------------------------------
# Finalize MPI
# -------------------------------------------------------------------

Short_MPI::MPI_Finalize();

1;
