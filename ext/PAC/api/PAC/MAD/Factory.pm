package PAC::MAD::Factory;

use strict;
use Carp;

use lib ("$ENV{UAL_PAC}/api/");
use Pac::Smf;

use PAC::MAD::Map;


sub new
{
    my ($type, $smf, $map) = @_;

    my $this = {};
    
    $this->{"smf"}   = $smf;
    $this->{"map"}   = $map;
    $this->{"drift"} = 0;

    bless $this, $type;
    $this->_initialize();

    return $this;  
}

# Elements

sub make_element
{
  my ($this, $name, $keyword) = @_;

  my ($element, $elem_key);
  my $flag = 0;

  # Check keyword

  my $keyword_4 = substr($keyword, 0, 4);
  $this->{"map"}->elemKeyFromString($keyword_4, \$elem_key); 

  if(defined $elem_key) { 
      $element = Pac::GenElement->new($name, $elem_key->key);
      $flag = 1;
  }
  else {
      my $iterator = $this->{smf}->elements->find($keyword);

      if( $iterator != $this->{smf}->elements->end()) {
	  $element = Pac::GenElement->new($name, $iterator->second->key);
	  $element->copy($iterator->second);      
	  $flag = 1;
      }
  }

  if($flag == 0) {
       $element = Pac::GenElement->new($name, $this->{drift}->key);
       print "PAC::MAD::Factory : $name, keyword ( $keyword ) ";
       print "is not a MAD keyword or element (replaced by drift)\n";
  }

  return $element;
}

sub _initialize
{
    my $this = shift;
    $this->{"drift"} = $this->{"map"}->elemKeyFromString("drif", \$this->{drift});
}

1;

__END__

=head1

=begin html
<h1> Class <a href="./package.html"> PAC::MAD</a>::Factory</h1>
<hr>
<h3> Extends: </h3>
The Factory class implements the operations to create SMF generic elements
according to a MAD element keyword or a MAD element class.
<hr>
<pre><h3>Sample Script:  <a href="./Factory.txt"> Factory.pl </a> </h3></pre>
<h3> Public Methods </h3>
<ul>
<li> <b> new($smf, $map) </b>
<dl> 
    <dt> Constructor.
    <dd><i>smf</i> - a pointer to a PAC::SMF instance.
    <dd><i>map</i> - a pointer to a <a href="./Map.html"> PAC::MAD::Map </a> instance.
</dl>
<li> <b> make_element($name, $keyword) </b>
<dl> 
    <dt> Creates a SMF generic element.
    <dd><i>name</i>    - a generic element name.
    <dd><i>keyword</i> - a MAD element type or a MAD element class (a 
	lower-case string, e.g., "sbend" )

</dl>
</ul> 
<hr>

=end html
