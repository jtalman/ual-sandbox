package UAL::MakeMaker;

use ExtUtils::MakeMaker;
use Carp;
use strict;

sub WriteMakefile {
    Carp::croak "WriteMakefile: Need even number of args" if @_ % 2;
    my %att = @_;
    $att{DEFINE}       = '-DHAS_BOOL -Ubool';
    $att{XSOPT}        = '-C++ -prototypes';
    $att{CC}           = "g++ -D_SYS_TIMES_H_";
    $att{LD}           = "g++";
    $att{CCCDLFLAGS}   = '-fPIC';
    $att{OPTIMIZE}     = '-g';
    ExtUtils::MakeMaker::WriteMakefile(%att);
}

1;

