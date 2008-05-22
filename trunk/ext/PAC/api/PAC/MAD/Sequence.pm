package PAC::MAD::Sequence;

use strict;
use Carp;

use lib ("$ENV{UAL_PAC}/api/");
use Pac::Smf;

use PAC::MAD::Helper;
use PAC::MAD::SequenceItem qw($SEQ_NAME $SEQ_ELEMENT $SEQ_AT $SEQ_ADD);

my $helper_;
my $SEQ_TINY = 1.e-10;

sub new 
{
  my ($type, $helper, $name, $attributes) = @_;

  my $this = {};

  # Data members

  $this->{name}  = $name;
  $this->{refer} = "centre";
  $this->{items} = [];

  # Global data

  $helper_ = $helper;

  # Restrictions

  if($attributes->{refer} ne "centre") { 
      croak "PAC::MAD::Sequence : this version supports only the attribute \"centre\" \n"; 
  } 

  return bless $this, $type;
}

sub set
{
   my $this = shift; 
   $this->remove();
   $this->add(@_);
}

sub add
{
  my $this = shift; 
  foreach (@_){ push @{$this->{items}}, PAC::MAD::SequenceItem->new($helper_->collection, @$_);}
}

sub remove 
{
  my $this = shift;
  $this->{items} = [];
}

sub lattice
{
  my ($this, $name) = @_;

  my ($l_key, $drift_key, $drift_id) = (0, 0, 0);
  $helper_->map->attribKeyFromString("l", \$l_key);
  $helper_->map->elemKeyFromString("drif", \$drift_key);
  $drift_id  = $drift_key->key;

  # make line

  my $line = Pac::Line->new($name);

  my ($item, $drift, $i, $length, $distance, $exit) = (0, 0, 0, 0.0, 0.0, 0.0);
  foreach $item ( @{$this->{items}}) {

        $length   = $item->[$SEQ_ELEMENT]->get($l_key);
        $distance = $item->[$SEQ_AT] - ($length/2. + $exit);

        if(abs($distance) < $SEQ_TINY) { $distance = 0.0; }


        if($distance >= 0) { 
	    $i++; 
	    $drift = Pac::GenElement->new("_d_$i", $drift_id);
	    $drift->set($distance*$l_key);
	}
	else {
	    croak "PAC::MAD::Sequence: the distance ($distance) before element " , 
	    $item->[$SEQ_NAME], " < 0 \n"; 
	}

        $line->add($drift);
	$line->add($item->[$SEQ_ELEMENT]);

        $exit = $item->[$SEQ_AT] + $length/2.;
  }

  # make lattice 

  my $lattice = Pac::Lattice->new($name);
  $lattice->set($line);

  # update element attributes

  my $adaptor;

  $i = 1;  
  foreach $item ( @{$this->{items}}) {
    if(defined $item->[$SEQ_ADD]) { 
	$adaptor = $helper_->registry->get_elemAdaptor($lattice->element($i)->key);
	$lattice->element($i)->setName($item->[$SEQ_NAME]);
	$adaptor->update_attributes($lattice->element($i), $item->[$SEQ_ADD]); 
    }
    $i += 2;
  }

  return $lattice;  
}

1;

__END__

=head1

=begin html
<h1> Class <a href="./package.html"> PAC::MAD</a>::Sequence</h1>
<hr>
<h3> Extends: </h3>
<p>
The Sequence class is an extension of the SMF data structures. It facilitates 
the description of a flat, fully-instantiated accelerator representation.
<hr>
<pre><h3>Sample Script:  <a href="./Sequence.txt"> Sequence.pl </a> </h3></pre>
<h3> Public Methods </h3>
<ul>
<li> <b> new($helper, $name, $attributes) </b>
<dl>
    <dt> Constructor.
    <dd><i>helper</i> - a pointer to a <a href="./Helper.html"> PAC::MAD::Helper </a> instance.
    <dd><i>name</i> - a sequence name.
    <dd><i>attributes</i> - a reference to a hash of MAD sequence attributes (e.g. {refer => centre}).
</dl>
<li> <b> set($items) </b>
<dl>
    <dt> Defines a sequence of lattice elements.
    <dd><i>items</i> - an array of references to MAD sequence items. Each item is represented by the
    following array of parameters: lattice element name, element type or  name of an element class,
    reference to a hash of item parameters (e.g. {at => 2.3}), reference to a hash of element attributes
    (e.g. {k1 => 0.004}).
    <dt> Examples:
    <dd><i>MAD 8</i> - s1 : sequence, refer = centre
    <dd><i>MAD 8</i> - qf1_0: qf1, at = 2.3, k1 = 0.004
    <dd><i>MAD 8</i> - . . . 
    <dd><i>MAD 8</i> - endsequence 
    <dd><i>Sequence</i> - $sequence = new PAC::MAD::Sequence($helper, "s1", {refer = > centre}); 
    <dd><i>Sequence</i> - $sequence->set(["qf1_0", "qf1", {at => 2.3}, {k1 => 0.004}], ...);
</dl>
<li> <b> add($items) </b>
<dl>
    <dt> Adds new sequence elements.
    <dd><i>items</i> - an array of references to MAD sequence items (see <i> set </i>).
</dl>
<li> <b> remove() </b>
<dl>
    <dt> Removes all sequence elements.
</dl>
<li> <b> lattice($name) </b>
<dl>
    <dt> Builds a new lattice.
    <dd><i>name</i> - a lattice name.
</dl>
</ul> 
<hr>

=end html
