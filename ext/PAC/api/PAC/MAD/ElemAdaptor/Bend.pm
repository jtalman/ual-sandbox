package PAC::MAD::ElemAdaptor::Bend;

use strict;
use Carp;
use vars qw(@ISA);

use PAC::MAD::Map;
use PAC::MAD::ElemAdaptor qw($ELEM_PI $ELEM_PART $ELEM_KEY $ELEM_ENTER $ELEM_BODY $ELEM_EXIT);
@ISA = qw(PAC::MAD::ElemAdaptor);

my @bend = ("angle", "fint");

# ***************************************************
# Public Interface
# ***************************************************

sub new
{
  my ($type, $map)  = @_;

  my $this  = PAC::MAD::ElemAdaptor->new($map);

  $this->{"keys"} = {};
  $this->{"ir"}    = 0;

  bless $this, $type;
  $this->_initialize($map);
  return $this;
}

sub update_attributes
{
   my ($this, $element, $attributes) = @_;

   my ($mad_key, $smf_key, $value, $key1, $order, $key2, $key, $ir);

   my $l = $attributes->{l};
   if(defined $l) { $l = $this->_update_attribute($element, $ELEM_BODY, $this->{l}, $l);}
   else           { $l = $element->get($this->{l}); } 
   if($l == 0) { croak "PAC::MAD::ElemAdaptor::Bend::update_attributes : ", $element->name, " l == 0 \n"; }

   my @attributes = %$attributes;

   while (@attributes) {
     $mad_key = shift @attributes; 
     $value   = shift @attributes;  
  
     if($value eq "tilt") { $value = $ELEM_PI/2.; }
     
     $smf_key = $this->{"keys"}->{$mad_key};

     if(not defined $smf_key) {
        if($value =~ /ir([0-9]*)/) { 
           $ir = 1;
           if($1 ne "") { $ir = $1; } 
   	   $this->_update_attribute($element, $ELEM_BODY, $this->{ir}, $ir);
        }
        else { 
		print "PAC::MAD::ElemAdaptor::Bend::update_attribute: ($mad_key) is not a SMF ",  
                                  $this->{keyword}, " attribute \n";
                next;
        }
     }
     else {
	 $this->_update_attribute($element, $smf_key->[$ELEM_PART], $smf_key->[$ELEM_KEY], $value);
     }
  }
  $this->_add_ends($element);

}

sub get_attribute
{ 
  my ($this, $element, $attribute) = @_;

  my $smf_key = $this->{"keys"}->{$attribute};
  my $value = 0;

  if(defined $smf_key) { 
    my $part  = $element->getPart($smf_key->[$ELEM_PART]);
    if($part != 0 ) { $value =  $part->get($smf_key->[$ELEM_KEY]); }
  }
  else {
    print "PAC::MAD::ElemAdaptor::Bend::get_attribute: ($attribute) is not a SMF ", 
           $this->{keyword}, " attribute \n";
  }
  return $value;
}

# ***************************************************
# "Protected" Interface
# ***************************************************

sub _initialize
{
  my ($this, $map) = @_;

  $this->{keyword} = "rbend";
  $map->attribKeyFromString("n", \$this->{ir});

  my @body = ("l", "angle", "fint", "tilt");

  my $attribKey = 0;

  foreach(@body) { 
      $map->attribKeyFromString($_, \$attribKey);
      $this->{"keys"}->{$_} = [$attribKey, $ELEM_BODY];
  };

  $map->attribKeyFromString("angle", \$attribKey);
  $this->{"keys"}->{"e1"} = [$attribKey, $ELEM_ENTER];
  $this->{"keys"}->{"e2"} = [$attribKey, $ELEM_EXIT];

}

sub _update_attribute
{
 my ($this, $element, $part, $key, $value) = @_;
 my $e_part;
 if($part == $ELEM_BODY)  { $e_part = $element->body; }
 else {
   if($part == $ELEM_ENTER) { $e_part = $element->front; }
   else { 
     if($part == $ELEM_EXIT)  { $e_part = $element->end; } 
     else { croak "PAC::MAD::ElemAdaptor::Bend::_update_attribute : element part($part) is out of [0,2] \n";}
   }
 }  

 my $delta = $value - $e_part->get($key);
 if($delta != 0) { $e_part->add($delta*$key); } 
 return $value;
}

sub _add_ends
{
}

sub _print_attributes
{
  my ($this, $element, $str) = @_;

  my $l  = $element->get($this->{l});
  if($l != 0 ) { $$str .= ", l = ". $l; }

  $this->_print_bend($element, $str);
  $this->_print_ends($element, $str);

  my $tilt  = $element->get($this->{"keys"}->{"tilt"}->[$ELEM_KEY]);
  if($tilt != 0 ) {
    if($tilt == $ELEM_PI/2.) { $$str .= " & \n, tilt "; }
    else                     { $$str .= " & \n, tilt = ". $tilt; }
  }
  my $ir = $element->get($this->{ir}); 

  if($ir != 0 ) {
    if($ir == 1) { $$str .= " & \n   , type = ir "; }
    else         { $$str .= " & \n   , type = ir". $ir; }
  }    

  $this->_print_mlt($element, $l, $str);

  $this->_print_aperture($element, $str);

}

sub _print_bend
{
  my ($this, $element, $str) = @_;

  my $value = 0;
  foreach(@bend) {
     $value  = $element->get($this->{"keys"}->{$_}->[$ELEM_KEY]);
     if($value != 0 ) { $$str .= ", " . $_ . " = ". $value; }
  }
}

sub _print_ends
{
}

sub _print_mlt
{
}

1;
