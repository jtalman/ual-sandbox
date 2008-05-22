use lib "$ENV{UAL}/env/$ENV{UAL_ARCH}";
use UAL::MakeMaker;

my $dir = "$ENV{UAL_ACCSIM}";

UAL::MakeMaker::WriteMakefile(
    'NAME'	   => 'Accsim::Bunch',
    'VERSION_FROM' => '../Accsim/Bunch.pm',
    'INST_LIB'     => "$dir/api/blib/$ENV{UAL_ARCH}/lib",
    'INST_ARCHLIB' => "$dir/api/blib/$ENV{UAL_ARCH}/arch",
    'INST_MAN3DIR' => "$dir/api/blib/$ENV{UAL_ARCH}/man3",
    'INC'          => "-I$dir/src -I$ENV{UAL_PAC}/src -I$ENV{UAL_CORE}/include",
    'LIBS'         => "-L$dir/lib/$ENV{UAL_ARCH} -lAccsim -L$ENV{UAL_PAC}/lib/$ENV{UAL_ARCH} -lPac -L$ENV{UAL_CORE}/lib/$ENV{UAL_ARCH} -lUal",
);
