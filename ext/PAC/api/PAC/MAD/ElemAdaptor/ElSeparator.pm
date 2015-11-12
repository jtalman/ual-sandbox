package PAC::MAD::ElemAdaptor::ElSeparator;

use strict;
use Carp;
use vars qw(@ISA);

use PAC::MAD::Map;

use PAC::MAD::ElemAdaptor qw($ELEM_PI);
@ISA = qw(PAC::MAD::ElemAdaptor);

# ***************************************************
# Public Interface
# ***************************************************

sub new
{
  my ($type, $map) = @_;
  my $this = PAC::MAD::ElemAdaptor->new($map);

  $this->{"keyword"} = "elseparator";
  $this->{"tilt"} = 0;
  $this->{"keys"} = ["l", "tilt"];

  $map->attribKeyFromString("tilt", \$this->{tilt});

  return bless $this, $type;
}

1;
