package UAL::ADXF::Parser;

use lib ("$ENV{UAL_EXTRA}/ADXF/api/blib/$ENV{UAL_ARCH}/arch", "$ENV{UAL_EXTRA}/ADXF/api/blib/$ENV{UAL_ARCH}/lib");

use strict;
use Carp;
use vars qw($VERSION @ISA @EXPORT $AUTOLOAD);

require Exporter;
require DynaLoader;
require AutoLoader;

@ISA = qw(Exporter DynaLoader);
@EXPORT = qw();
$VERSION = '1.00';

bootstrap UAL::ADXF::Parser $VERSION;

1;
__END__

=head1 NAME

UAL::ADXF::Parser -  extension for blah blah blah

=head1 SYNOPSIS

  use Da:Tps;
  blah blah blah

=head1 DESCRIPTION

Blah blah blah.

=head1 AUTHOR

Nikolay Malitsky

=head1 SEE ALSO

Blah blah blah..

=cut
