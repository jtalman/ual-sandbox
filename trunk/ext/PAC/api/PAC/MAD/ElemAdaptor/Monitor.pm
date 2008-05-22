package PAC::MAD::ElemAdaptor::Monitor;

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

  $this->{"keyword"} = "monitor";
  $this->{"keys"}  = ["l"];

  return bless $this, $type;
}

1;
