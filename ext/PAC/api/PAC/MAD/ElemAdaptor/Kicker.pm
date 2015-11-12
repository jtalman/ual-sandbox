package PAC::MAD::ElemAdaptor::Kicker;

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

  $this->{"keyword"} = "kicker";
  $this->{"hkick"} = 0;
  $this->{"vkick"} = 0;
  $this->{"tilt"}  = 0;
  $this->{"keys"}  = ["l", "hkick", "vkick", "tilt"];

  bless $this, $type;
  $this->_initialize($map);
  return $this;
}

sub _initialize
{
    my ($this, $map) = @_;

    my ($mult, $kl, $ktl) = (0, 0, 0);
    $map->bucketKeyFromString("mult", \$mult);   
    $map->attribKeyFromString("kl",  \$kl); 
    $map->attribKeyFromString("ktl", \$ktl);  

    $this->{hkick} = $mult->attribKey($kl->index, 0);  
    $this->{vkick} = $mult->attribKey($ktl->index, 0); 

    $map->attribKeyFromString("tilt", \$this->{tilt});
 
}

1;
