package PAC::MAD::ElemAdaptor::SinglePole;

use strict;
use Carp;
use vars qw(@ISA);

use POSIX;

use PAC::MAD::Map;

use PAC::MAD::ElemAdaptor qw($ELEM_PI);
@ISA = qw(PAC::MAD::ElemAdaptor);

# ***************************************************
# Public Interface
# ***************************************************

my $strengthKey_ = 0;

sub new
{
  my ($type, $map, $order) = @_;
  my $this = PAC::MAD::ElemAdaptor->new($map);

  $this->{k}     = 0;
  $this->{tilt}  = 0;
  $this->{ir}    = 0;

  bless $this, $type;
  $this->_initialize($map, $order);
  return $this;
}

sub update_attributes
{
   my ($this, $element, $attributes) = @_;

   my $l = $attributes->{l};
   if(defined $l) { $l = $this->_update_attribute($element, $this->{l}, $l);}
   else           { $l = $element->get($this->{l}); } 
   if($l == 0)    { croak $this->{keyword}, " ::update_attributes : ", $element->name, " l == 0 \n"; }

   my $k = $attributes->{ $this->_k };
   if(defined $k) { $this->_update_attribute($element, $this->{$this->_k}, $k*$l/$this->_factor); }

   my $tilt = $attributes->{tilt};
   if(defined $tilt) { 
   	if($tilt eq "tilt") { $tilt = $ELEM_PI/$this->_tilt; }   
   	$this->_update_attribute($element, $this->{tilt}, -1*$tilt);
   }  
   my $ir  =  $attributes->{type};
   if(defined $ir) { 
   	if($ir =~ /ir([0-9]*)/) { 
           $ir = 1;
          if($1 ne "") { $ir = $1; }  
   	  $this->_update_attribute($element, $this->{ir}, $ir);
        }
   }     
}

sub get_attribute
{
  my ($this, $element, $attribute) = @_;
  my $key = $this->{$attribute};
  my $value;
  if(defined $key) { $value = $element->get($key); }
  else {
     croak $this->{keyword}, " ::get_attribute: ($attribute) is not a ", $this->{keyword}, " attribute \n";
  }
  if($attribute eq $this->_k){
      $value /= $element->get($this->{l})/$this->_factor;
  }
  return $value;  
}

# ***************************************************
# "Protected" Interface
# ***************************************************

sub _key
{
    return $strengthKey_;
}

sub _k
{
}

sub _tilt
{
}

sub _factor
{
}

sub _print_attributes
{
   my ($this, $element, $str) = @_;

   my $l  = $element->get($this->{l});
   if($l != 0 ) { 
       $$str .= ", l = " . $l;

       my $k = $element->get($this->{$this->_k})/$l*$this->_factor;
       if($k != 0 ) { $$str.= ", " . $this->_k . " = " . $k; }
   }

   my $tilt = $element->get($this->{tilt});
   if($tilt != 0 ) {
    if(abs($tilt) == $ELEM_PI/$this->_tilt) { $$str .= ", &\n tilt "; }
    else                                    { $$str .= ", &\n tilt = ". $tilt; }
   }
   my $ir = $element->get($this->{ir}); 
   if($ir != 0 ) {
    if($ir == 1) { $$str .= ", &\n type = ir "; }
    else         { $$str .= ", &\n type = ir". $ir; }
   }    

   $this->_print_aperture($element, $str);
  
}

sub _initialize
{
    my ($this, $map, $order) = @_;

    my ($mult, $kl) = (0, 0);
    $map->bucketKeyFromString("mult", \$mult);   
    $map->attribKeyFromString("kl", \$kl);

    $strengthKey_ = $mult->attribKey($kl->index, $order);

    $map->attribKeyFromString("tilt", \$this->{tilt});
    $map->attribKeyFromString("n",    \$this->{ir}); 

}

1;

