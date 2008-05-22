package PAC::MAD::ElemAdaptor::Rcollimator;

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

  $this->{"keyword"} = "rcollimator";
  $this->{"xsize"} = 0;
  $this->{"ysize"} = 0;
  $this->{"keys"}  = ["l", "xsize", "ysize"];

  $map->attribKeyFromString("xsize", \$this->{xsize});
  $map->attribKeyFromString("ysize", \$this->{ysize});

  return bless $this, $type;
}

1;
