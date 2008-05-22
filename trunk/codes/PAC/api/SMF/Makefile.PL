use lib "$ENV{UAL}/env/$ENV{UAL_ARCH}";
use UAL::MakeMaker;

my $dir = "$ENV{UAL_PAC}";

UAL::MakeMaker::WriteMakefile(
    'NAME'	   => 'Pac::Smf',
    'VERSION_FROM' => '../Pac/Smf.pm',
    'INST_LIB'     => "$dir/api/blib/$ENV{UAL_ARCH}/lib",
    'INST_ARCHLIB' => "$dir/api/blib/$ENV{UAL_ARCH}/arch",
    'INST_MAN3DIR' => "$dir/api/blib/$ENV{UAL_ARCH}/man3",
    'INC'          => "-I$dir/src -I$ENV{UAL_ZLIB}/src -I$ENV{UAL_CORE}/include",
    'LIBS'         => "-L$dir/lib/$ENV{UAL_ARCH} -lPacSMF -lPacOptics -lPac -L$ENV{UAL_ZLIB}/lib/$ENV{UAL_ARCH} -lZTps -L$ENV{UAL_CORE}/lib/$ENV{UAL_ARCH} -lUal",
);
