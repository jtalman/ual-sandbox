package PAC::MAD::ElemAdaptor::Solenoid;

use strict;
use Carp;
use vars qw(@ISA);

use PAC::MAD::Map;

use PAC::MAD::ElemAdaptor;
@ISA = qw(PAC::MAD::ElemAdaptor);

# ***************************************************
# Public Interface
# ***************************************************

sub new
{
  my ($type, $map) = @_;
  my $this = PAC::MAD::ElemAdaptor->new($map);

  $this->{"keyword"}  = "solenoid";
  $this->{"ks"}    = 0;
  $this->{"keys"}  = ["l", "ks"];

  $map->attribKeyFromString("ks", \$this->{ks});

  return bless $this, $type;
}

1;
