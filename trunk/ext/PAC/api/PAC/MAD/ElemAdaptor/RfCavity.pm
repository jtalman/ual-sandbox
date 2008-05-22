package PAC::MAD::ElemAdaptor::RfCavity;

use strict;
use Carp;
use vars qw(@ISA);

use PAC::MAD::Helper;

use PAC::MAD::ElemAdaptor qw($ELEM_PI);
@ISA = qw(PAC::MAD::ElemAdaptor);

# ***************************************************
# Public Interface
# ***************************************************

my @body = ("volt", "lag", "harmon");

sub new
{
  my ($type, $map)  = @_;
  my $this  = PAC::MAD::ElemAdaptor->new($map);

  $this->{"keyword"} = "rfcavity";
  $this->{"keys"} = ["l", "volt", "lag", "harmon"];

  bless $this, $type;
  $this->_initialize($map);
  return $this;
}

# ***************************************************
# "Protected" Interface
# ***************************************************

sub _initialize
{
  my ($this, $map) = @_;

  my ($bucketKey, $attribKey) = (0, 0);
  $map->bucketKeyFromString("rfca", \$bucketKey);

  foreach(@body) { 
      $map->attribKeyFromString($_, \$attribKey);
      $this->{$_} = $bucketKey->attribKey($attribKey->index, 0);
  };

}

sub update_attributes
{
  my ($this, $element, $attributes) = @_;

  if(not defined($attributes)) { return; }
  my @attributes = %$attributes;

  my ($mad_key, $value, $smf_key);

  while(@attributes) {

    $mad_key = shift @attributes;
    $value   = shift @attributes;
    $smf_key = $this->{$mad_key};

    if(not defined $smf_key) {
      print "PAC::MAD::ElemAdaptor::update_attribute: ($mad_key) is not a SMF ",  
               $this->{keyword}, " attribute \n";
    }   
    else {
        if($mad_key eq "volt") { $value /= 1000.; }
	$this->_update_attribute($element, $smf_key, $value);
    }

  }
}

sub _print_attributes
{
  my ($this, $element, $string) = @_;

  my $value = 0.0;
  foreach(@{$this->{"keys"}}) {
     $value  = $element->get($this->{$_});
     if($value != 0 ) { 
	if($_ eq "harmon"){ $$string .= ", freq = ". $value; }
	else { $$string .= ", " . $_ . " = ". $value; }
     }
  }
}

1;

