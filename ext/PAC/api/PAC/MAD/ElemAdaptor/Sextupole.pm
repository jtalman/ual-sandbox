package PAC::MAD::ElemAdaptor::Sextupole;

use strict;
use Carp;
use vars qw(@ISA);

use PAC::MAD::ElemAdaptor::SinglePole;
@ISA = qw(PAC::MAD::ElemAdaptor::SinglePole);

# ***************************************************
# Public Interface
# ***************************************************

sub new
{
  my ($type, $map) = @_;
  my $this = PAC::MAD::ElemAdaptor::SinglePole->new($map, 2);

  $this->{keyword}  = "sextupole";
  $this->{k2}       = $this->_key;

  return bless $this, $type;
}

# ***************************************************
# "Protected" Interface
# ***************************************************

sub _k
{
    return "k2";
}

sub _tilt
{
    return 6;
}

sub _factor
{
    return 2;
}

1;

