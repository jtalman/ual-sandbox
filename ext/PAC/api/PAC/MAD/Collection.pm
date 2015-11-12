package PAC::MAD::Collection;

use strict;
use Carp;

use lib ("$ENV{UAL_PAC}/api/");
use Pac::Smf;

use PAC::MAD::Map;

my $map_;

sub new
{
    my ($type, $smf, $map) = @_;

    my $this = {};    
    $this->{smf}   = $smf;
    $this->{items} = {};

    $map_ = $map;

    return bless $this, $type;
}

sub get_element
{
  my ($this, $keyword) = @_;

  my ($element, $keyword4, $name, $elemKey);
  my $iterator = $this->{smf}->elements->find($keyword);

  if( $iterator != $this->{smf}->elements->end()) {
      $element = $iterator->second;   
  }
  else {
      $keyword4 = substr($keyword, 0, 4);
      $element = $this->{items}->{$keyword4};
      if(not defined $element) {
	  $map_->elemKeyFromString($keyword4, \$elemKey);
	  if(defined $elemKey){
	      $name = "__" . $keyword4 . "_";
	      $element =  Pac::GenElement->new($name, $elemKey->key);
	      $this->{items}->{$keyword4} = $element;
	  }
	  else{
	      croak "PAC::MAD::Collection::get_element(\$keyword): ", 
	      "$keyword is not an element name or an element type \n"; 
	  }	      
      }
  }

  return $element;
}


1;

__END__


=head1

=begin html
<h1> Class <a href="./package.html"> PAC::MAD</a>::Collection</h1>
<hr>
<h3> Extends: </h3>
The Collection class implements a simple  interface to query the SMF element collection
by a MAD keyword (element class or element type).
<hr>
<pre><h3>Sample Script:  <a href="./Collection.txt"> Collection.pl </a> </h3></pre>
<h3> Public Methods </h3>
<ul>
<li> <b> new($smf, $registry) </b>
<dl> 
    <dt> Constructor.
    <dd><i>smf</i> - a pointer to a PAC::SMF instance.
    <dd><i>registry</i> - a pointer to a <a href="./Registry.html"> PAC::MAD::Registry </a> instance.
</dl>
<li> <b> get_element($keyword) </b>
<dl> 
    <dt> Returns a pointer to a SMF generic element selected by a MAD keyword. 
    <dd><i>keyword</i> - a MAD element class or a MAD element type (a lower case string ).
</dl>
</ul> 
<hr>

=end html
