package Da::Integrator;

use strict;
use Carp;

use Da::Const qw($X_ $PX_ $Y_ $PY_ $CT_ $DE_ $PROTON_ $INFINITY_);

sub new
{
  my $type = shift;
  my $this = {}; 
  return bless $this, $type; 
}

sub propagate
{
  my ($this, $position, $beam_att) = @_;

  # Define v0byc & charge from beam attributes

  $this->{mass}   = $PROTON_    unless defined($this->{mass}   =  $beam_att->{MASS});
  $this->{charge} = 1.          unless defined($this->{charge} =  $beam_att->{CHARGE});
  $this->{energy} = $INFINITY_  unless defined($this->{energy} =  $beam_att->{ENERGY});

  $this->{v0byc}  = $this->v0byc();

  #  Represent the object $position as a Perl array
  #  of its coordinates to decouple the external and
  #  Perl interfaces

  my $p = [];
  $this->split($position, $p); 

  #  Transform coordinates to the new frame
 
  $this->transform($p);

  #  Update the $position coordinates

  $this->join($position, $p);
}

sub transform
{
  my ($this, $p) = @_;
}

sub psp0
{
   my ($this, $p) = @_;

   my $size = @$p;
   if($size < $PX_) { return; }

   my $psp0  = 1. - $p->[$PX_]*$p->[$PX_];

   if($size > $PY_){ 
	$psp0 -= $p->[$PY_]*$p->[$PY_]; 
   } 
   if($size > $DE_){
        $psp0 += $p->[$DE_]*$p->[$DE_];
        $psp0 += (2./$this->{v0byc})*$p->[$DE_];
   }

   $psp0 = sqrt($psp0);	

   return $psp0;
}

sub split
{
  my ($this, $position, $p) = @_;

  my $k;   
  for($k=0; $k < $position->size; $k++) {
        push @$p,   $position->value($k) + 0.0; 
  }
}

sub join
{
  my ($this, $position, $p) = @_;

  my $k;
  for($k=0; $k < $position->size; $k++) {
    $position->value($k, $p->[$k] + 0.0);
  }
}

sub v0byc
{
  my $this= shift;
  my $e0 = $this->{energy};
  my $m0 = $this->{mass};
  my $v  = sqrt($e0*$e0 - $m0*$m0)/$e0;
  return $v;
}


1;
