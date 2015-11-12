package PAC::MAD::ElemAdaptor::Vkicker;

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

  $this->{"keyword"}  = "vkick";
  $this->{"kick"}  = 0;
  $this->{"tilt"}  = 0;
  $this->{"keys"}  = ["l", "kick", "tilt"];

  bless $this, $type;
  $this->_initialize($map);
  return $this;
}

sub _initialize
{
    my ($this, $map) = @_;

    my ($mult, $ktl) = (0, 0);
    $map->bucketKeyFromString("mult", \$mult);   
    $map->attribKeyFromString("ktl",  \$ktl); 
 
    $this->{kick} = $mult->attribKey($ktl->index, 0);  

    $map->attribKeyFromString("tilt", \$this->{tilt});
 
}

1;
