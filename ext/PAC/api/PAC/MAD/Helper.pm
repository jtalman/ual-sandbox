package PAC::MAD::Helper;

use strict;
use Carp;

use PAC::MAD::Factory;
use PAC::MAD::Collection;
use PAC::MAD::Registry;
use PAC::MAD::Map;

my $factory_;
my $collection_;
my $registry_;
my $map_;

sub new 
{
    my ($type, $smf) = @_;

    my $this = {};

    $map_        = new PAC::MAD::Map($smf);
    $registry_   = new PAC::MAD::Registry($map_);
    $factory_    = new PAC::MAD::Factory($smf, $map_);
    $collection_ = new PAC::MAD::Collection($smf, $map_);

    return bless $this, $type; 
}

sub factory
{
    my $this = shift;
    return $factory_;

}

sub collection
{
    my $this = shift;
    return $collection_;
}

sub registry
{
    my $this = shift;
    return $registry_;
}

sub map
{
    my $this = shift;
    return $map_;
}

1;

__END__

=head1

=begin html
<h1> Class <a href="./package.html"> PAC::MAD</a>::Helper</h1>
<hr>
<h3> Extends: </h3>
The Helper class is a container that encapsulates several services to 
facilitate the MAD-to-SMF migration. These services  are 
<a href="./Factory.html"> PAC::MAD::Factory </a>, 
<a href="./Collection.html"> PAC::MAD::Collection </a>, 
<a href="./Registry.html"> PAC::MAD::Registry </a>, 
and
<a href="./Map.html"> PAC::MAD::Map </a>. 
<hr>
<pre><h3>Sample Script:  <a href="./Helper.txt"> Helper.pl </a> </h3></pre>
<h3> Public Methods </h3>
<ul>
<li> <b> new($smf) </b>
<dl>
    <dt> Constructor.
    <dd><i>smf</i> - a pointer to a SMF instance.
</dl>
<li> <b> factory() </b>
<dl>
    <dt>Returns a pointer to a <a href="./Factory.html"> PAC::MAD::Factory </a> instance.
</dl>
<li> <b> collection() </b>
<dl>
    <dt>Returns a pointer to a <a href="./Collection.html"> PAC::MAD::Collection </a> instance.
</dl>
<li> <b> registry() </b>
<dl>
    <dt>Returns a pointer to a <a href="./Registry.html"> PAC::MAD::Registry </a> instance.
</dl>
<li> <b> map() </b>
<dl>
    <dt>Returns a pointer to a <a href="./Map.html"> PAC::MAD::Map </a> instance.
</dl>
</ul> 
<hr>

=end html
