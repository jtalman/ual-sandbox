package PAC::MAD::Registry;

use strict;
use Carp;

use PAC::MAD::Map;

use PAC::MAD::ElemAdaptor::Marker;
use PAC::MAD::ElemAdaptor::Drift;
use PAC::MAD::ElemAdaptor::Sbend;
use PAC::MAD::ElemAdaptor::Rbend;
use PAC::MAD::ElemAdaptor::Quadrupole;
use PAC::MAD::ElemAdaptor::Sextupole;
use PAC::MAD::ElemAdaptor::Octupole;
use PAC::MAD::ElemAdaptor::Multipole;
use PAC::MAD::ElemAdaptor::Solenoid;
use PAC::MAD::ElemAdaptor::Hkicker;
use PAC::MAD::ElemAdaptor::Vkicker;
use PAC::MAD::ElemAdaptor::Kicker;
use PAC::MAD::ElemAdaptor::RfCavity;
use PAC::MAD::ElemAdaptor::ElSeparator;
use PAC::MAD::ElemAdaptor::Hmonitor;
use PAC::MAD::ElemAdaptor::Vmonitor;
use PAC::MAD::ElemAdaptor::Monitor;
use PAC::MAD::ElemAdaptor::Ecollimator;
use PAC::MAD::ElemAdaptor::Rcollimator;

# use PAC::MAD::Multipole;

sub new
{
    my ($type, $map) = @_;

    my $this = {};
    $this->{items} = {};

    bless $this, $type;
    $this->_initialize($map);

    return $this;
}

sub get_elemAdaptor
{
    my ($this, $elemKeyID) = @_;
    return $this->{items}->{$elemKeyID};
}

sub _initialize
{
  my ($this, $map) = @_;

  my $elemKey;

  $map->elemKeyFromString("mark", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Marker->new($map);
  $map->elemKeyFromString("drif", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Drift->new($map);
  $map->elemKeyFromString("sben", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Sbend->new($map);
  $map->elemKeyFromString("rben", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Rbend->new($map);
  $map->elemKeyFromString("quad", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Quadrupole->new($map);
  $map->elemKeyFromString("sext", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Sextupole->new($map);
  $map->elemKeyFromString("octu", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Octupole->new($map);
  $map->elemKeyFromString("mult", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Multipole->new($map);
  $map->elemKeyFromString("sole", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Solenoid->new($map);
  $map->elemKeyFromString("hkic", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Hkicker->new($map);
  $map->elemKeyFromString("vkic", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Vkicker->new($map);
  $map->elemKeyFromString("kick", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Kicker->new($map);
  $map->elemKeyFromString("rfca", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::RfCavity->new($map);
  $map->elemKeyFromString("else", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::ElSeparator->new($map);
  $map->elemKeyFromString("hmon", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Hmonitor->new($map);
  $map->elemKeyFromString("vmon", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Vmonitor->new($map);
  $map->elemKeyFromString("moni", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Monitor->new($map);
  $map->elemKeyFromString("ecol", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Ecollimator->new($map);
  $map->elemKeyFromString("rcol", \$elemKey);
  $this->{items}->{$elemKey->key}  = PAC::MAD::ElemAdaptor::Rcollimator->new($map);


#  $this->{items}->{$map->elemKeyFromString(mult)->key}  = PAC::MAD::Multipole->new($map);

}

1;

__END__

=head1

=begin html
<h1> Class <a href="./package.html"> PAC::MAD</a>::Registry</h1>
<hr>
<h3> Extends: </h3>
The Registry class contains adaptors for all MAD element types.
<hr>
<pre><h3>Sample Script:  <a href="./Registry.txt"> Registry.pl </a> </h3></pre>
<h3> Public Methods </h3>
<ul>
<li> <b> new($map) </b>
<dl> 
    <dt> Constructor. 
    <dd><i>map</i> - a pointer to a <a href="./Map.html"> PAC::MAD::Map </a> instance. 
</dl>
<li> <b> get_elemAdaptor($elemKeyID) </b>
<dl> 
    <dt>Returns an adaptor for a concrete element type selected by a SMF element ID.
    <dd><i>elemKeyID</i> - a SMF element ID.
</dl>
</ul> 
<hr>

=end html  
