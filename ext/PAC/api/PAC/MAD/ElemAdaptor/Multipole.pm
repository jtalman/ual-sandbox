package PAC::MAD::ElemAdaptor::Multipole;

use strict;
use Carp;
use vars qw(@ISA);

use PAC::MAD::Map;

use PAC::MAD::ElemAdaptor qw($ELEM_PI $ELEM_BODY);
@ISA = qw(PAC::MAD::ElemAdaptor);

# ***************************************************
# Public Interface
# ***************************************************

my $MAX_ORDER = 9;

sub new
{
  my ($type, $map) = @_;
  my $this = PAC::MAD::ElemAdaptor->new($map);

  $this->{keyword} = "multipole";
  $this->{mult} = 0;
  $this->{kl}   = [];
  $this->{ktl}  = [];
  $this->{factor} = [];

  bless $this, $type;
  $this->_initialize($map);
  return $this;
}

sub update_attributes
{
  my ($this, $element, $attributes) = @_;

  if(not defined($attributes)) { return; }

  my ($mad_key, $value, $order, $smf_key);

  my @attributes = %$attributes;

  while(@attributes) {
    $mad_key = shift @attributes;
    $value   = shift @attributes;
    if($mad_key =~/(k)([0-9]+).*/){
      $order = $2;
      $smf_key = $this->{kl}->[$order];

      if(not defined $smf_key) {
         print "PAC::MAD::ElemAdaptor::Multipole::update_attribute: ($mad_key) is not a SMF ",  
               $this->{keyword}, " attribute \n";
         next;
      }   

      $this->_update_attribute($element, $smf_key, $value/$this->{factor}->[$order]);
    }
    else{
      print "PAC::MAD::ElemAdaptor::Multipole::update_attribute: ($mad_key) is not a SMF ",  
               $this->{keyword}, " attribute \n";
    }
  }
  $element->add(0.0*$this->{l});
}

sub get_attribute
{
  my ($this, $element, $mad_key) = @_;
  my $value = 0.0;
  my ($order, $smf_key);
  if($mad_key =~/(k)([0-9]+).*/){
	$order = $2;
        $smf_key = $this->{kl}->[$order];
        $value =  $element->get($smf_key)*$this->{factor}->[$order]; 
  }
  else {
    croak $this->{keyword}, " ::get_attribute: ($mad_key) is not a ", $this->{keyword}, " attribute \n";
  }
  return $value;
}

sub _print_attributes
{
   my ($this, $element, $string) = @_;

   my $mlt = $this->_get_bucket($element, $ELEM_BODY, $this->{mult});

   my ($size, $order, $io, $value);
   if($mlt) {
      $size  = $this->{mult}->size;
      $order = $mlt->size/$size;

      for($io = 0; $io < $order; $io++){
           $value = $mlt->value($this->{kl}->[$io]->index);
           if($value) { $$string .= ", k" . $io ."l = " . $value; }
     }
   }
   
}

sub _initialize
{
  my ($this, $map) = @_;

  $map->bucketKeyFromString("mult", \$this->{mult});

  my ($kl, $ktl) = (0, 0);

  $map->attribKeyFromString("kl", \$kl);  
  $map->attribKeyFromString("ktl", \$ktl);  

  my $i;

  $this->{kl}->[0]      = $this->{mult}->attribKey($kl->index, 0);
  $this->{ktl}->[0]     = $this->{mult}->attribKey($ktl->index, 0);
  $this->{factor}->[0]  = 1;

  for($i = 1; $i < $MAX_ORDER; $i++){
    $this->{kl}->[$i]       = $this->{mult}->attribKey($kl->index, $i);
    $this->{ktl}->[$i]      = $this->{mult}->attribKey($ktl->index, $i);
    $this->{factor}->[$i]   = $this->{factor}->[$i-1]*$i;
  }
}

1;
