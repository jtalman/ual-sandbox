package Accsim::Bunch;

use lib ("$ENV{UAL_ACCSIM}/api/blib/$ENV{UAL_ARCH}/arch", 
         "$ENV{UAL_ACCSIM}/api/blib/$ENV{UAL_ARCH}/lib",
         "$ENV{UAL_PAC}/api/");

use strict;
use Carp;
use vars qw($VERSION @ISA @EXPORT $AUTOLOAD);

require Exporter;
require DynaLoader;
require AutoLoader;

@ISA = qw(Exporter DynaLoader);

@EXPORT = qw();
$VERSION = '1.00';

bootstrap Accsim::Bunch $VERSION;

package Accsim::Bunch;

1;
