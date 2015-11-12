package PAC::MAD::SequenceItem;

use PAC::MAD::Collection;

use vars qw(@ISA @EXPORT_OK $SEQ_NAME $SEQ_ELEMENT $SEQ_AT $SEQ_ADD);

require Exporter;
@ISA = qw(Exporter);
@EXPORT_OK = qw($SEQ_NAME $SEQ_ELEMENT $SEQ_AT $SEQ_ADD);

*SEQ_NAME    = \0;
*SEQ_ELEMENT = \1;
*SEQ_AT      = \2;
*SEQ_ADD     = \3;

use strict;
use Carp;

# Constructor

# collection - a reference to PAC::MAD::Collection
# name       - a lattice element name
# keyword    - an element keyword or a name of a MAD element class
# params     - MAD sequence node parameters (e.g. at)
# attributes - element deviations

sub new
{
  my ($type, $collection, $name, $keyword, $params, $attributes) = @_;
  my $this = [];

  $this->[$SEQ_NAME]    = $name;
  $this->[$SEQ_ELEMENT] = 0;
  $this->[$SEQ_AT]      = 0.0;
  $this->[$SEQ_ADD]     = $attributes if defined $attributes;

  # Define a generic element

  my $element = $collection->get_element($keyword);
  if(defined $element) {   
      $this->[$SEQ_ELEMENT] = $element
  }
  else{
      croak "PAC::MAD::SequenceItem::new() : $keyword is not defined \n";
  }

  # Define an element position

  if (defined $params) {
      $this->[$SEQ_AT]  =  $params->{at} if defined $params->{at}; 
  }

  return bless $this, $type;

}

1;


