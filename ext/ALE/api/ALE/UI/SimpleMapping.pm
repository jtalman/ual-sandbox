package ALE::UI::SimpleMapping;

use strict;
use Carp;

my $DIM = 6;

sub new
{
    my $type = shift;
    my $this = {};
    return bless $this, $type;
}

sub space
{
    my ($this, $shell, $order) = @_;

    if(not ref $shell->{"space"}) {
	if(defined $order) {
	    $shell->{space} = new Zlib::Space($DIM, $order);
	} 
	else {
	    croak "ALE::UI::SimpleMapping::make_space: the DA space is not defined \n";
	} 
    } 
}

sub map
{
    my ($this, $shell, $order, $de) = @_;

    my $beam = $shell->{"beam"};

    my $map = new Pac::TMap($DIM);

    # $map->refOrbit($shell->{"orbit"});
    my $orbit = new Pac::Position(); 
    $shell->{"code"}->clorbit($orbit, $beam);  
    $orbit->de($de);
    $map->refOrbit($orbit);
    
    $shell->{"code"}->map($map, $beam, $order);  
    $shell->{"map"} = $map;   
}

1;

__END__


=head1

=begin html
<h1> Class <a href="./package.html"> ALE::UI</a>::SimpleMapping</h1>
<hr>
<h3> Extends: </h3>
The SimpleMapping class implements several operations with a one-turn map. 
<hr>
<h3> Public Methods </h3>
<ul>
<li> <b> new() </b>
<dl>
    <dt> Constructor.
</dl>
<li> <b> space($shell, $order) </b>
<dl>
    <dt> Defines global parameters of 6D maps. 
    <dd><i>shell</i>   - a pointer to an ALE::UI::Shell instance. 
    <dd><i>order</i>   - the maximum order of Taylor maps.  
</dl>
<li> <b> map($shell, $order) </b>
<dl>
    <dt> Makes a one-turn 6D map. 
    <dd><i>shell</i>   - a pointer to an ALE::UI::Shell instance. 
    <dd><i>order</i>   - the map order.  
</dl>
</ul> 
<hr>

=end html
