package LHC::Teapot::Smf;

use strict;
use Carp;
use vars qw(@ISA);

use lib ("$ENV{UAL_MAD}/api/");
use Mad::Smf;
@ISA = qw(Mad::Smf);

sub new
{
  my $type   = shift; 
  my %params = @_;

  my $this = Mad::Smf->new();
  return  bless $this, $type;
}

sub store
{
  my $this = shift;
  my %params = @_;
  my $mad   = $params{file};

  open mad, ">$mad" or die "Can' find the MAD file $mad \n";

  # Elements

  my ($it_e, $element, $holder, $str, $bname, $l, $angle, $k1l);

  for($it_e = $this->elements->begin(); $it_e != $this->elements->end(); $it_e++){
     $element = $it_e->second;
     $holder  = $this->_helper->{elements}->{$element->key};

     if($holder->{type} eq "sbend") { 

         $angle = $element->get($this->_helper->{attrib_keys}->{angle});
         $l     = $element->get($this->_helper->{attrib_keys}->{l});
         $k1l   = -( sin(0.5*$angle)/cos(0.5*$angle) )/($l/$angle);
         $bname  = $this->_reduce_name($element->name());
         
         printf mad "i$bname : multipole, k1l = $k1l \n";
         printf mad "o$bname : multipole, k1l = $k1l \n";
         my $n = 16;
         printf mad "b$bname : sbend, l = $l/$n, angle = $angle/$n \n";  
         printf mad "$bname  : line = (i$bname, $n*b$bname, o$bname) \n";       
     }
     else{
         printf mad $holder->print($element, $this->_reduce_name($element->name()));
     }

  } # elements

  # Lattices

  printf mad "\n\n";

  my ($it_l, $lattice, $counter, $name);
  for($it_l = $this->lattices->begin(); $it_l != $this->lattices->end(); $it_l++){

        $lattice = $it_l->second;
        printf mad $lattice->name . " : line = ( & \n"; 

        $name = $this->_reduce_name($lattice->element(0)->genName());
        printf mad sprintf("  %-8s", $name );
        $counter = 1;

        for($it_e = 1; $it_e < $lattice->size(); $it_e++) {
           $name = $this->_reduce_name($lattice->element($it_e)->genName());
           printf mad sprintf(", %-8s", $name );

           $counter ++;
           if($counter == 6) { printf mad " & \n"; $counter = 0;}     
        }
        printf mad ") \n";
  }
  close(mad);
}

sub _reduce_name
{
  my ($this, $name) = @_;
  $name =~ tr/_//d;
  $name =~ s/(qs1)(qd|qf)(.*)/$2$3/;
  return $name;
}

1;