use lib "$ENV{UAL_MPI_PERL}/config/$ENV{UAL_ARCH}";

use Short_MPI::MakeMaker;

Short_MPI::MakeMaker::WriteMakefile(
    'NAME'	   => 'Short_MPI',
    'LIBS'         => "-lmpich",
    'OBJECT'       => 'Short_MPI_wrap.o'
);
