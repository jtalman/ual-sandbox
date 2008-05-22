use lib "$ENV{UAL}/env/$ENV{UAL_ARCH}"; 
use UAL::MakeMaker;

my $dir = "$ENV{UAL_SXF}";
my $ual_arch = $ENV{UAL_ARCH};

UAL::MakeMaker::WriteMakefile(
    'NAME'	   => 'UAL::SXF::Parser',
    'VERSION_FROM' => '../../../UAL/SXF/Parser.pm',
    'INST_LIB'     => "$dir/api/blib/$ENV{UAL_ARCH}/lib",
    'INST_ARCHLIB' => "$dir/api/blib/$ENV{UAL_ARCH}/arch",
    'INST_MAN3DIR' => "$dir/api/blib/$ENV{UAL_ARCH}/man3",
    'INC'          => " -I$dir/src -I$ENV{UAL_ZLIB}/src -I$ENV{UAL_PAC}/src " .
		      " -I$ENV{UAL_CORE}/include -I$ENV{SXF}/src ",
    'LIBS'         => " -L$dir/lib/$ual_arch -L$ENV{SXF}/lib/$ual_arch " .
		      " -L$ENV{UAL_ZLIB}/lib/$ual_arch -L$ENV{UAL_PAC}/lib/$ual_arch " .
		      " -L$ENV{UAL_CORE}/lib/$ual_arch -lUal " .
		      " -lUalSXF2 -lSXF -lPacSMF -lPacOptics -lPac -lZTps",
);
