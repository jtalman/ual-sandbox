package PAC::FTPOT::Shell;

use strict;
use Carp;
use vars qw(@ISA);

use PAC::MAD::Shell;
@ISA = qw(PAC::MAD::Shell);

my $cuts_ = [];

my ($sbendID_, $lKey_, $angleKey_) = (0, 0, 0);

sub new
{
    my ($type, $smf) = @_;
    
    my $this = new PAC::MAD::Shell($smf);

    bless $this, $type;
    $this->_initialize();
    return $this;
}

sub cut
{
    my ($this, $pattern, $replacement) = @_;
    push @{$cuts_}, [$pattern, $replacement];
}

sub split
{
    my ($this, $pattern, $n) = @_;

    my $n_key = 0;
    $this->_helper->map->attribKeyFromString("n", \$n_key );

    my $it = 0;
    my $smf = $this->_smf;
    for($it = $smf->elements->begin(); $it != $smf->elements->end(); $it++){
	if($it->second->name =~ $pattern) { $it->second->add($n*$n_key); }
    }
}

sub save
{
  my ($this, $file) = @_;

  open(MAD, ">$file");

  # Elements

  my ($it_e, $element, $adaptor, $short_name);

  my $smf    = $this->_smf;
  my $helper = $this->_helper;

  for($it_e = $smf->elements->begin(); $it_e != $smf->elements->end(); $it_e++){
     $element = $it_e->second;
     $adaptor = $helper->registry->get_elemAdaptor($element->key);

     if(not defined $adaptor){
	 croak "PAC::FTPOT::Shell::save: element ", $element->name , 
	 ", its type (", $element->key, ") is not supported \n";
     }

     if($element->key == $sbendID_) { 
	 $this->_print_sbend(\*MAD, $adaptor, $element); 
     }
     else {
	 $this->_print_gen_element(\*MAD, $adaptor, $element);
     } 
  } 

  # Lattices

  printf MAD "\n\n";

  my ($it_l, $lattice, $counter, $name);
  for($it_l = $smf->lattices->begin(); $it_l != $smf->lattices->end(); $it_l++){

        $lattice = $it_l->second;
        printf MAD $lattice->name . " : line = ( & \n"; 

        $short_name = sprintf "%s", $lattice->element(0)->genName;
	$this->_cut(\$short_name);
        printf MAD sprintf("  %-8s", $short_name);

        $counter = 1;

        for($it_e = 1; $it_e < $lattice->size(); $it_e++) {
	    $counter = $this->_print_latt_element(\*MAD, $lattice->element($it_e), $counter); 
        }
        printf MAD ") \n";
  }  

  close(MAD);
}

sub _initialize
{
    my $this = shift;
    my $helper = $this->_helper;

    $helper->map->attribKeyFromString("angle", \$angleKey_);
    $helper->map->attribKeyFromString("l", \$lKey_);  

    my $sbendKey;
    $helper->map->elemKeyFromString("sben", \$sbendKey);
    $sbendID_ = $sbendKey->key;

}

sub _cut
{
    my ($this, $name) = @_;
    foreach(@$cuts_){ $$name =~ s/$_->[0]/$_->[1]/g; }
}


sub _print_sbend
{
    my ($this, $mad, $adaptor, $element) = @_;
    
    my $bname = sprintf "%s", $element->name;
    $this->_cut(\$bname);

    my $front = $element->getPart(0);
    my $end = $element->getPart(2);

#    if( $front != 0 || $end != 0){
     if(defined $front) {
	my $l     = $element->get($lKey_);
	if($l == 0) {
	    croak "PAC::FTPOT::Shell::_print_sbend: $element->name - length == 0 \n";
	}

	my $e1 = $element->front->get($angleKey_);
	my $e2 = $element->end->get($angleKey_);

	my $e1k1l   = -( sin(0.5*$e1)/cos(0.5*$e1) )*$e1/($l);
	my $e2k1l   = -( sin(0.5*$e2)/cos(0.5*$e2) )*$e2/($l);
         
	printf $mad "el$bname : multipole, k1l = $e1k1l \n";
	printf $mad "er$bname : multipole, k1l = $e2k1l \n";

    }

    printf $mad $adaptor->get_string($element, $bname);
}

sub _print_gen_element
{
    my ($this, $mad, $adaptor, $element) = @_;

    my $short_name = sprintf "%s", $element->name;
    $this->_cut(\$short_name); 

    my $front = $element->getPart(0);
    my $end = $element->getPart(2);

#    if( $front != 0 || $end != 0){
    if(defined $front) {
	printf $mad   "el$short_name : multipole \n";
	printf $mad   "er$short_name : multipole \n";
    }
  
    printf $mad $adaptor->get_string($element, $short_name);
}

sub _print_latt_element
{
    my ($this, $mad, $element, $counter) = @_;

    my $short_name = sprintf "%s", $element->genName;
    $this->_cut(\$short_name);

    my $front = $element->getPart(0);
    my $end   = $element->getPart(2);

#   if($front != 0 || $end != 0) {
    if(defined $front) {
	$counter = $this->_print_trio($mad, \$short_name, $counter);
    }
    else {
	printf $mad sprintf(", %-8s",$short_name );
	$counter ++;
	if($counter == 6) { printf $mad " & \n"; $counter = 0;}     
    }

    return $counter;
}    

sub _print_trio
{
   my ($this, $mad, $short_name, $counter) = @_;  

   printf $mad sprintf(", el%-7s", $$short_name);
   $counter ++;
   if($counter == 6) { printf $mad " & \n"; $counter = 0;}     
   printf $mad sprintf(", %-8s", $$short_name);
   $counter ++;
   if($counter == 6) { printf $mad " & \n"; $counter = 0;} 
   printf $mad sprintf(", er%-7s", $$short_name);
   $counter ++;
   if($counter == 6) { printf $mad " & \n"; $counter = 0;} 

   return $counter;
}

1;

__END__

=head1

=begin html
<h1> Class <a href="./package.html"> PAC::FTPOT</a>::Shell</h1>
<hr>
<h3> Extends: <a href="../MAD/Shell.html"> PAC::MAD::Shell </a> </h3>
The Shell class represents the FTPOT-specific user-friendly interface to 
SMF data structures.
<hr>
<pre><h3>Sample Script:  <a href="./Shell.txt"> Shell.pl </a> </h3></pre>
<h3> Public Methods </h3>
<ul>
<li> <b> split($pattern, $ir) </b>
<dl>
    <dt> Selects generic elements according to a pattern (given by a 
	regular expression) and specifies into how many thin multipoles 
	the elements will be divided in TEAPOT algorithms. 
    <dd><i>pattern</i> - a regular expression for selecting generic elements.
    <dd><i>ir</i> - a TEAPOT split number (1 - IR, 2 - IR2, etc.).
    <dt> Examples:
    <dd><i>FTPOT</i> - qf1 : quadrupole, ..., type = ir2
    <dd><i>FTPOT</i> - qd1 : quadrupole, ..., type = ir2
    <dd><i>Shell</i> - split("^q(f|d)1", 2)
</dl>
<li> <b> cut($pattern, $replacement) </b>
<dl>
    <dt> Selects generic elements and builds an  additional short name 
    according to the Perl substitution operator: s/$pattern/$replacement/g.
    <dd><i>pattern</i> - a pattern expression.
    <dd><i>replacement</i> - a replacement expression.
    <dt> Examples:
    <dd>cut("_", "") - remove all underscores in element names.
</dl>
</ul> 
<hr>

=end html
