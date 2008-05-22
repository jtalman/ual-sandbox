package Short_MPI;

use lib ("$ENV{UAL_MPI_PERL}/api/blib/$ENV{UAL_ARCH}/arch",
	 "$ENV{UAL_MPI_PERL}/api/blib/$ENV{UAL_ARCH}/lib");

use vars qw($VERSION @ISA @EXPORT $AUTOLOAD);

require Exporter;
require DynaLoader;
require AutoLoader;

@ISA = qw(Exporter DynaLoader);

@EXPORT = qw( MPI_Init MPI_Finalize 
              MPI_Comm_rank 
              MPI_Comm_size 
              MPI_Wtime 
              MPI_Barrier 
	      MPI_Initialized );

$VERSION = '1.00';

bootstrap Short_MPI $VERSION;

1;
