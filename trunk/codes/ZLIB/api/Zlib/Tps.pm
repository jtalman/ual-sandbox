package Zlib::Tps;

use lib ("$ENV{UAL_ZLIB}/api/blib/$ENV{UAL_ARCH}/arch", "$ENV{UAL_ZLIB}/api/blib/$ENV{UAL_ARCH}/lib");

use strict;
use Carp;
use vars qw($VERSION @ISA @EXPORT $AUTOLOAD);

require Exporter;
require DynaLoader;
require AutoLoader;

@ISA = qw(Exporter DynaLoader);
@EXPORT = qw();
$VERSION = '1.00';

bootstrap Zlib::Tps $VERSION;

package Zlib::Space;

package Zlib::Tps;
@Zlib::Tps::ISA = qw(Zlib::Space);

use overload
	"+" => \&add,
	"*" => \&multiply,
        "-" => \&subtract,
        "/" => \&divide,
        "sqrt" => \&sqrt;


package Zlib::VTps;
@Zlib::VTps::ISA = qw(Zlib::Space);

use overload
	"+" => \&add,
	"*" => \&multiply,
        "-" => \&subtract,
        "/" => \&divide;

package Zlib::Tps;

1;
__END__

=head1 NAME

Zlib::Tps - Perl

=cut
