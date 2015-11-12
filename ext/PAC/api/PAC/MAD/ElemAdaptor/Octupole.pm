package PAC::MAD::ElemAdaptor::Octupole;

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
  my $this = PAC::MAD::ElemAdaptor::SinglePole->new($map, 3);

  $this->{keyword}  = "quadrupole";
  $this->{k3}       = $this->_key;

  return bless $this, $type;
}

# ***************************************************
# "Protected" Interface
# ***************************************************

sub _k
{
    return "k3";
}

sub _tilt
{
    return 8;
}

sub _factor
{
    return 6; 
}

1;

