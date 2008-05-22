use lib "$ENV{UAL}/env/$ENV{UAL_ARCH}"; 
use UAL::MakeMaker;

my $dir = "$ENV{UAL_ZLIB}";

UAL::MakeMaker::WriteMakefile(
    'NAME'	   => 'Zlib::Tps',
    'VERSION_FROM' => '../Zlib/Tps.pm',			      
    'INST_LIB'     => "$dir/api/blib/$ENV{UAL_ARCH}/lib/",
    'INST_ARCHLIB' => "$dir/api/blib/$ENV{UAL_ARCH}/arch/",
    'INST_MAN3DIR' => "$dir/api/blib/man3/",
    'INC'          => "-I$dir/src ",
    'LIBS'         => "-L$dir/lib/$ENV{UAL_ARCH} -lZTps",
);
