package PAC::MAD::ElemAdaptor::Quadrupole;

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
  my $this = PAC::MAD::ElemAdaptor::SinglePole->new($map, 1);

  $this->{keyword}  = "quadrupole";
  $this->{k1}       = $this->_key;

  return bless $this, $type;
}

# ***************************************************
# "Protected" Interface
# ***************************************************

sub _k
{
    return "k1";
}

sub _tilt
{
    return 4;
}

sub _factor
{
    return 1;
}

1;


