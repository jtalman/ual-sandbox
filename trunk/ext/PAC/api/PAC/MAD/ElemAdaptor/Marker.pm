package PAC::MAD::ElemAdaptor::Marker;

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
  $this->{"keyword"} = "marker";
  $this->{"keys"} = [];
  return bless $this, $type;
}

1;

