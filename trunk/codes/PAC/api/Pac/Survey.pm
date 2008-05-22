package Pac::Survey;

use lib ("$ENV{UAL_PAC}/api/blib/$ENV{UAL_ARCH}/arch", "$ENV{UAL_PAC}/api/blib/$ENV{UAL_ARCH}/lib");

use strict;
use Carp;
use vars qw($VERSION @ISA @EXPORT $AUTOLOAD);

require Exporter;
require DynaLoader;
require AutoLoader;

@ISA = qw(Exporter DynaLoader);
@EXPORT = qw();
$VERSION = '1.00';

bootstrap Pac::Survey $VERSION;

1;
__END__
