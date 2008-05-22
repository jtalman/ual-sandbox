package ALE::UI::SimpleMatrix;

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
	    $shell->{space} = new Da::Space($DIM, $order);
	} 
	else {
	    croak "ALE::UI::SimpleMatrix::space: the DA space is not defined \n";
	} 
    } 
}

sub map
{
    my ($this, $shell, $order, $delta) = @_;

    my $map = new Pac::TMap($DIM);
    # $map->refOrbit($shell->{"orbit"});
    
    my $mltOrder =  $map->mltOrder();

    $map->mltOrder($order);
    $shell->{"code"}->matrix($map, $shell->{"beam"}, $delta);
    $map->mltOrder($mltOrder);    

    $shell->{"map"} = $map;   
}

1;

__END__


=head1

=begin html
<h1> Class <a href="./package.html"> ALE::UI</a>::SimpleMatrix</h1>
<hr>
<h3> Extends: </h3>
The SimpleMatrix class implements several operations with a one-turn matrix. 
<hr>
<h3> Public Methods </h3>
<ul>
<li> <b> new() </b>
<dl>
    <dt> Constructor.
</dl>
<li> <b> map($shell) </b>
<dl>
    <dt> Makes a one-turn 6D matrix. 
    <dd><i>shell</i>   - a pointer to an ALE::UI::Shell instance.  
</dl>
</ul> 
<hr>

=end html
