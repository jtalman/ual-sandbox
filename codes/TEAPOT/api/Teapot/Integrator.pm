package Teapot::Integrator;

use lib ("$ENV{UAL_TEAPOT}/api/blib/$ENV{UAL_ARCH}/arch", 
         "$ENV{UAL_TEAPOT}/api/blib/$ENV{UAL_ARCH}/lib",
         "$ENV{UAL_PAC}/api/blib/$ENV{UAL_ARCH}/arch",
         "$ENV{UAL_PAC}/api/blib/$ENV{UAL_ARCH}/lib",
         "$ENV{UAL_PAC}/api/");

use strict;
use Carp;
use vars qw($VERSION @ISA @EXPORT $AUTOLOAD);

require Exporter;
require DynaLoader;
require AutoLoader;

use Pac::Smf;

@ISA = qw(Exporter DynaLoader);

@EXPORT = qw();
$VERSION = '1.00';

bootstrap Teapot::Integrator $VERSION;

@Teapot::Element::ISA = qw(Pac::LattElement);

package Teapot::Integrator;

1;
__END__
