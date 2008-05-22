package Short_MPI::MakeMaker;

use ExtUtils::MakeMaker;
use Carp;
use strict;

sub WriteMakefile {
    Carp::croak "WriteMakefile: Need even number of args" if @_ % 2;
    my %att = @_;
    my $dir = "$ENV{UAL_MPI_PERL}";
    $att{INST_LIB}     = "$dir/api/blib/$ENV{UAL_ARCH}/lib";
    $att{INST_ARCHLIB} = "$dir/api/blib/$ENV{UAL_ARCH}/arch";
    $att{INST_MAN3DIR} = "$dir/api/blib/$ENV{UAL_ARCH}/man3";
    $att{INC}          = "-I$ENV{MPIHOME}/include $att{INC}";
    $att{LIBS}         = "-L$ENV{MPIHOME}/lib/shared  $att{LIBS}";
    $att{XSOPT}        = '-C++ -prototypes';
    $att{CC}           = "g++  -D_SYS_TIMES_H";
    $att{LD}           = "g++ ";
    ExtUtils::MakeMaker::WriteMakefile(%att);
}

1;
