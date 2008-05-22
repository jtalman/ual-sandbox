package PAC::MAD::ElemAdaptor;

use strict;
use Carp;
use vars qw(@ISA @EXPORT_OK $ELEM_KEY $ELEM_PART $ELEM_ENTER $ELEM_BODY $ELEM_EXIT $ELEM_PI);

require Exporter;
@ISA = qw(Exporter);
@EXPORT_OK = qw($ELEM_KEY $ELEM_PART $ELEM_ENTER $ELEM_BODY $ELEM_EXIT $ELEM_PI); 

*ELEM_KEY  = \0;
*ELEM_PART = \1;

*ELEM_ENTER = \0;
*ELEM_BODY  = \1;
*ELEM_EXIT  = \2;

*ELEM_PI     = \3.1415926536;

use PAC::MAD::Map;

sub new
{
  my ($type, $map) = @_;

  my $this = {};
  $this->{keyword} = "";
  $this->{l}       = 0;
  $this->{shape}   = 0;
  $this->{xsize}   = 0;
  $this->{ysize}   = 0;
  $map->attribKeyFromString("l", \$this->{l});
  $map->attribKeyFromString("shape", \$this->{shape});
  $map->attribKeyFromString("xsize", \$this->{xsize});
  $map->attribKeyFromString("ysize", \$this->{ysize});
  return bless $this, $type;
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
	$this->_update_attribute($element, $smf_key, $value);
    }

  }
}

sub get_attribute
{
  my ($this, $element, $attribute) = @_;

  my $smf_key = $this->{$attribute};
  my $value   = 0;

  if(defined $smf_key) { 
    $value =  $element->get($smf_key);
  }
  else {
    print "PAC::MAD::ElemAdaptor::get_attribute: ($attribute) is not a SMF ", 
           $this->{keyword}, " attribute \n";
  }
  return $value;
}

sub get_string
{
  my ($this, $element, $name) = @_;
  my $str = "";
  $this->_print_attributes($element, \$str );
  return sprintf("%-8s : %-10s %s\n", $name, $this->{keyword}, $str);
}

# ***************************************************
# "Protected" Interface
# ***************************************************

sub _get_bucket
{
   my ($this, $smf_element, $part, $bucket) = @_;
   my $p = $smf_element->getPart($part);
#   if($p == 0) { return 0; }
   if(not defined $p) { return 0; }
   my $attributes = $p->attributes;
   if($attributes != 0){
 	my $ib;  
   	for($ib = $attributes->begin(); $ib != $attributes->end(); $ib++){
       		if($ib->second->key == $bucket->key) { return $ib->second; } 
     	}
   }
   return 0;   
}

sub _update_attribute
{
 my ($this, $element, $key, $value) = @_;
 my $delta = $value - $element->get($key);
 if($delta != 0) { $element->add($delta*$key); } 
 return $value;
}

sub _print_attributes
{
  my ($this, $element, $string) = @_;

  my $value = 0.0;
  foreach(@{$this->{"keys"}}) {
     $value  = $element->get($this->{$_});
     if($value != 0 ) { $$string .= ", " . $_ . " = ". $value; }
  }
  $this->_print_aperture($element, $string);
}

sub _print_aperture
{
  my ($this, $element, $string) = @_;
  my $ual_keys = ["shape", "xsize", "ysize"];
  my $tpot_keys = ["typeaper", "xapsize", "yapsize"];
  my ($counter, $idk) = (0, 0);
  my $value = 0;
  foreach(@{$ual_keys}) {
     $value  = $element->get($this->{$_});
     if($value != 0 ) { 
	if($counter == 0) {
	  $$string .= ", &\n " . $tpot_keys->[$idk] . " = ". $value; $counter++;
        }
	else {
          $$string .= ",  " . $tpot_keys->[$idk] . " = ". $value; 
        }
     }
     $idk++;
  }
}

sub _print_length
{
  my ($this, $element, $string) = @_;
  my $l = $element->get($this->{l});
  if($l != 0 ) { $$string = ", l = " . $l; }
  else         { $$string = "";}
}


1;

__END__


=head1

=begin html
<h1> Class <a href="./package.html"> PAC::MAD</a>::ElemAdaptor</h1>
<hr>
<h3> Extends: </h3>
The ElemAdaptor specifies the general interface to access SMF generic element 
attributes from <a href="./Shell.html"> PAC::MAD::Shell </a>. The adaptors
for each element type are  implemented via the corresponding derived classes.
<hr>
<pre><h3>Sample Script:  <a href="./ElemAdaptor.txt"> ElemAdaptor.pl </a> </h3></pre>
<h3> Public Methods </h3>
<ul>
<li> <b> new($map) </b>
<dl> 
    <dt> Constructor. 
    <dd><i>map</i> - a pointer to a <a href="./Map.html"> PAC::MAD::Map </a> instance. 
</dl>
<li> <b> update_attributes($element, $attributes) </b>
<dl> 
    <dt>Updates element attributes (adds new or replaces old ones).
    <dd><i>element</i> - a pointer to a PAC::SMF::GenElement.
    <dd><i>attributes</i> - a pointer to a hash of MAD element attributes (e.g. {l => 1.2, k1 => 3.5}).
</dl>
<li> <b> get_attribute($element, $attribute) </b>
<dl> 
    <dt> Returns an attribute value.
    <dd><i>element</i> - a pointer to a PAC::SMF::GenElement.
    <dd><i>attribute</i> - a MAD element attribute (specified by a 
	lower-case string, e.g. "k1" or "l").
</dl>
<li> <b> get_string($element, $name) </b>
<dl> 
    <dt> Writes element parameters to a string.
    <dd><i>element</i> - a pointer to a PAC::SMF::GenElement.
    <dd><i>name</i> - an additional element name provided by the user.
</dl>
</ul> 
<hr>

=end html  

