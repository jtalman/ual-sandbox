package PAC::MAD::ElemAdaptor::Sbend;

use strict;
use Carp;
use vars qw(@ISA);

use PAC::MAD::Map;
use PAC::MAD::ElemAdaptor qw($ELEM_PI $ELEM_PART $ELEM_KEY $ELEM_ENTER $ELEM_BODY $ELEM_EXIT);
use PAC::MAD::ElemAdaptor::Bend;
@ISA = qw(PAC::MAD::ElemAdaptor::Bend);

# ***************************************************
# Public Interface
# ***************************************************

my @ends = ("e1", "e2");

sub new
{
  my ($type, $map)  = @_;
  my $this  = PAC::MAD::ElemAdaptor::Bend->new($map);
  $this->{keyword} = "sbend";
  return bless $this, $type;
}

# ***************************************************
# "Protected" Interface
# ***************************************************


sub _print_ends
{
  my ($this, $element, $str) = @_;
 
  my ($part, $value) = (0, 0);
  foreach(@ends) {
     $part   = $element->getPart($this->{"keys"}->{$_}->[$ELEM_PART]);
     if($part != 0){
       $value  = $part->get($this->{"keys"}->{$_}->[$ELEM_KEY]);
       if($value != 0 ) { $$str .= " & \n, " . $_ . " = ". $value; }
     }
  }

}

1;
