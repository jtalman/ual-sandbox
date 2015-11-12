package PAC::MAD::Shell;

use strict;
use Carp;
use vars qw($shell $sequence $pi $twopi);

use POSIX;
use File::Basename;

# SMF stuff

use lib ("$ENV{UAL_PAC}/api/");
use Pac::Smf;

use PAC::MAD::Parser;
use PAC::MAD::Helper;
use PAC::MAD::Sequence;

my $smf_ = 0;
my $helper_ = 0;
my $parser_ = 0;
my $sequences_ = {};

my $fcounter_ = 0;

sub new
{
    my ($type, $smf) = @_;
    
    my $this = {};

    $smf_     = $smf;
    $parser_  = new PAC::MAD::Parser();
    $helper_  = new PAC::MAD::Helper($smf);

    bless $this, $type;
    return $this;
}

sub attribute
{
    my ($this, $element_name, $attribute) = @_;

    my $value = 0.0;

    my ($element, $adaptor);
    $element = $helper_->collection->get_element($element_name);

    if(defined $element) { 
	$adaptor = $helper_->registry->get_elemAdaptor($element->key);
	$ value = $adaptor->get_attribute($element, $attribute);
    }
    return $value;
}

sub element
{
    my ($this, $name, $keyword, $attributes) = @_;

    my $element = $helper_->factory->make_element($name, $keyword);
    my $adaptor = $helper_->registry->get_elemAdaptor($element->key);

    if(defined $adaptor) { 
	$adaptor->update_attributes($element, $attributes);
    }
    else {
	croak "PAC::MAD::Shell::element: element $name, $keyword type is not supported \n";
    }

    return $element;
}

sub line
{
  my $this = shift;
  my $name = shift;

  my $line = Pac::Line->new($name);

  my ($it, @lines);
  foreach (@_) {

      $it = $smf_->lines->find($_);

      if($it != $smf_->lines->end()) { 
	  $line->add($it->second); 
      }
      else{
	  $it = $smf_->elements->find($_);
	  if($it != $smf_->elements->end()) { 
	      $line->add($it->second); 
	  }
	  else { 
	      croak "PAC::MAD::Shell::line : $_ is not an element or a line \n";
	  }
      }
  }
  return $line;
}

sub lattice
{
    my ($this, $name, $src) = @_;

    my ($flag, $lattice) = (0, 0);
    my $it = $smf_->lines->find($src);

    if($it != $smf_->lines->end()) { 
	$lattice = Pac::Lattice->new($name);
	$lattice->set($it->second); 
	$flag = 1;
    }
    else {
	my @keys = keys %$sequences_;
	foreach(@keys){ 

	    if($_ eq $src) { 
		my $sequence = $sequences_->{$_}; 
		$lattice = $sequence->lattice($name);
		$flag = 1;
		last;
	    } 
	}
    }

    if( $flag == 0) { 
	croak "PAC::MAD::Shell::lattice : $name is not a line or a sequence\n"; 
    }

    return $lattice;
}

sub sequence
{
  my ($this, $name, $attributes) = @_;

  return $sequences_->{$name} = PAC::MAD::Sequence->new($helper_, $name, $attributes);;
}

sub call
{
    my $this = shift;
    $this->_call(@_);
}

sub save
{
  my ($this, $file) = @_;
  open (MAD, ">$file"); 

  # Elements

  my ($it_e, $element, $adaptor);

  for($it_e = $smf_->elements->begin(); $it_e != $smf_->elements->end(); $it_e++){
     $element = $it_e->second;
     $adaptor = $helper_->registry->get_elemAdaptor($element->key);

     if (defined  $adaptor){
	 printf MAD $adaptor->get_string($element, $element->name);
     }
     else {
	 croak "PAC::MAD::Shell::save: element ", $element->name , 
	 ", its type (", $element->key, ") is not supported \n";
     }
  } 

  # Lattices

  printf MAD "\n\n";

  my ($it_l, $lattice, $counter, $name);
  for($it_l = $smf_->lattices->begin(); $it_l != $smf_->lattices->end(); $it_l++){

        $lattice = $it_l->second;
        printf MAD $lattice->name . " : line = ( & \n"; 

        printf MAD sprintf("  %-8s", $lattice->element(0)->genName);
        $counter = 1;

        for($it_e = 1; $it_e < $lattice->size(); $it_e++) {
           printf MAD sprintf(", %-8s",$lattice->element($it_e)->genName );

           $counter ++;
           if($counter == 6) { printf MAD " & \n"; $counter = 0;}     
        }
        printf MAD ") \n";
  }  

  close(MAD);
}

sub smf
{
    return $smf_;
}

sub map
{
    return $helper_->map;
}

sub _call
{
    my $this = shift;

    my $id = shift;

    my $files;
    foreach (@_) { push @$files, $_; }

    $fcounter_++;
    my $basename = basename($files->[0]);

    my $script = $basename . "__smf" . $fcounter_ . "_" . $id . ".pl"; 

    local $shell = $this;
    local $sequence = 0; ;
    local $pi    = 2 * atan2(1,1)*2;
    local $twopi = 2* $pi;

    $parser_->translate("files" => $files, "script" => $script, "id" => $id);

    require "$script";

    unlink $script;    

}

sub _smf
{
    return $smf_;
}

sub _helper
{
    return $helper_;
}

sub _sequences
{
    return $sequences_;
}

1;

__END__


=head1

=begin html
<h1> Class <a href="./package.html"> PAC::MAD</a>::Shell</h1>
<hr>
<h3> Extends: </h3>
The Shell class represents the MAD-specific user-friendly interface to 
SMF data structures.
<hr>
<pre><h3> Sample Script:  <a href="./Shell.txt"> Shell.pl </a> </h3></pre>
<h3> Public Methods </h3>
<ul>
<li> <b> new($smf) </b>
<dl>
    <dt> Constructor.
    <dd><i>smf</i> - a pointer to a SMF instance.
</dl>
<li> <b> attribute($label, $attribute) </b>
<dl>
    <dt>Returns  a value of the MAD element attribute.
    <dd><i>label</i> - an element name. 
    <dd><i>attribute</i> - a MAD element attribute. 
    <dt>Examples:
    <dd><i>MAD 8</i> - q1[k1]
    <dd><i>Shell</i> - attribute("q1", "k1")
</dl>
<li> <b> element($label, $keyword, $attributes) </b>
<dl>
    <dt> Builds a new  element. 
    <dd><i>label</i> - an element name. 
    <dd><i>keyword</i> - a MAD element type or a MAD element class. 
    <dd><i>attributes</i> - a reference to a hash of the MAD element attributes. 
    <dt>Examples:
    <dd><i>MAD 8</i> - q1: quadrupole, l = 2.3, k1 = 0.3e-5
    <dd><i>MAD 8</i> - q2: q1
    <dd><i>Shell</i> - element("q1", "quadrupole", {l => 2.3, k1 => 0.3e-5})
    <dd><i>Shell</i> - element("q2", "q1", {})
</dl>
<li> <b> line($label, $members) </b>
<dl>
    <dt> Builds a new beam line.
    <dd><i>label</i> - a line  name. 
    <dd><i>members</i> - an  array of line members, lines and generic elements.
    <dt>Examples:
    <dd><i>MAD 8</i> - li1: line = (q1, q2)
    <dd><i>MAD 8</i> - li2: line = (li1, q1, li1)
    <dd><i>Shell</i> - line("li1", "q1", "q2")
    <dd><i>Shell</i> - line("li2", "li1", "q1", "li1")        
</dl>
<li> <b> sequence($label, $attributes) </b>
<dl>
    <dt> Builds a new sequence.
    <dd><i>label</i> - a sequence  name. 
    <dd><i>attributes</i> - a reference to a hash of MAD sequence attributes.
    <dt>Examples:
    <dd><i>MAD 8</i> - se1: sequence, refer=centre
    <dd><i>Shell</i> - sequence("se1", {refer => "centre"})     
</dl>
<li> <b> call(@files) </b>
<dl>
    <dt> Reads MAD input files.
    <dd><i>files</i> - an array of file names. 
    <dt>Examples:
    <dd><i>MAD 8</i> - call, filename="file1"
    <dd><i>MAD 8</i> - call, filename="file2"
    <dd><i>Shell</i> - call("file1", "file2")    
</dl>
<li> <b> save($file) </b>
<dl>
    <dt> Writes a MAD input file.
    <dd><i>file</i> - a file name. 
    <dt>Examples:
    <dd><i>MAD 8</i> - save, filename="file1"
    <dd><i>Shell</i> - save("file1")    
</dl>
<li> <b> lattice($name, $src) </b>
<dl>
    <dt> Builds a new lattice from a beam line or a sequence of elements.
    <dd><i>name</i> - a lattice name.
    <dd><i>src</i> -  a line name or a sequence name.
</dl>
<li> <b> smf() </b> 
<dl>
    <dt> Returns a pointer to a SMF instance
</dl>
<li> <b> map() </b>
<dl>
    <dt> Returns a pointer to a <a href="./Map.html"> PAC::MAD::Map </a> instance
</dl>
</ul> 
<hr>

=end html







