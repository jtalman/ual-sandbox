package Teapot::Math;

use lib ("$ENV{UAL_TEAPOT}/api/blib/$ENV{UAL_ARCH}/arch", 
         "$ENV{UAL_TEAPOT}/api/blib/$ENV{UAL_ARCH}/lib");

use strict;
use Carp;
use vars qw($VERSION @ISA @EXPORT $AUTOLOAD);

require Exporter;
require DynaLoader;
require AutoLoader;

@ISA = qw(Exporter DynaLoader);

@EXPORT = qw();
$VERSION = '1.00';

bootstrap Teapot::Math $VERSION;

package Teapot::RandomGenerator;

package Teapot::Math;

1;
