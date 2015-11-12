package PAC::MAD::ElemAdaptor::Hkicker;

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

  $this->{"keyword"}  = "hkick";
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

    my ($mult, $kl) = (0, 0);
    $map->bucketKeyFromString("mult", \$mult);   
    $map->attribKeyFromString("kl",  \$kl); 
 
    $this->{kick} = $mult->attribKey($kl->index, 0);  

    $map->attribKeyFromString("tilt", \$this->{tilt});
 
}

1;
