package PAC::MAD::ElemAdaptor::Rbend;

use strict;
use Carp;
use vars qw(@ISA);

use PAC::MAD::Map;
use PAC::MAD::ElemAdaptor qw($ELEM_PI $ELEM_PART $ELEM_KEY $ELEM_ENTER $ELEM_BODY $ELEM_EXIT);
use PAC::MAD::ElemAdaptor::Bend;
@ISA = qw(PAC::MAD::ElemAdaptor::Bend);

# ***************************************************
# Public Interface
# ***************************************************

sub new
{
  my ($type, $map)  = @_;
  my $this  = PAC::MAD::ElemAdaptor::Bend->new($map);
  $this->{keyword} = "rbend";
  return bless $this, $type;
}

1;

