package Da::Lie::Integrator;

use strict;
use Carp;

use Da::Const qw($X_ $PX_ $Y_ $PY_ $CT_ $DE_ $PROTON_ $INFINITY_);
use Zlib::Tps;

sub new
{
  my $type   = shift;
  my %params = @_;

  my $this = {};
 
  $this->{L}      = $params{L};
  $this->{N}      = $params{N};

  return bless $this, $type;
}


sub propagate
{
  my ($this, $object, $beam_att) = @_;

  my $morder =  $this->{N};
  if($morder <= 0) { croak "Da::Lie::Integrator: N, max. order of power series, <= 0 \n"; }

  $this->{mass}   = $PROTON_    unless defined($this->{mass}   =  $beam_att->{MASS});
  $this->{charge} = 1.          unless defined($this->{charge} =  $beam_att->{CHARGE});
  $this->{energy} = $INFINITY_  unless defined($this->{energy} =  $beam_att->{ENERGY});
  $this->{v0byc}  = $this->v0byc();

  my $h = $this->hamiltonian($object) + 0.0;

  my $i;
  my $tmp = $object + 0.0;
  my $sum = $object + 0.0;
  for($i = 1; $i <= $morder; $i++){
     $tmp = $h->vpoisson($tmp)/$i;
     $sum  += $tmp;
  }
  for($i =0; $i < $object->size; $i++) { $object->value($i, $sum->value($i)); }
  $object->order($object->order);
}

sub hamiltonian
{
  my ($this, $p) = @_;

  my $h = 1.;
  if($p->size < $PX_) { return $h;}

  # Beam

  my $v0byc  = $this->{v0byc}; 
  my $charge = $this->{charge}; 

  my $p0  = new Zlib::VTps($p->size);
  $p0 += 1;

  # Potential 

  my ($a0, $ax, $ay, $as) = $this->aPotential($p0);

  $ax *= $charge;
  $ay *= $charge;
  $as *= $charge;

  # 

  my $tmp = $p0->value($PX_) - $ax;
  $h = 1. - $tmp*$tmp;

  if($p0->size > $PY_) { 
        $tmp = $p0->value($PY_) - $ay;
	$h -= $tmp*$tmp;
  }

  if($p0->size > $DE_) { 
     $h += $p0->value($DE_)*$p0->value($DE_);
     $h += (2./$v0byc)*$p0->value($DE_);
  }

  $h = sqrt($h);
  $h *= 1. + $a0*$p0->value($X_);
  $h += $as;

  if($p0->size > $DE_) { $h -=  $p0->value($DE_)/$v0byc; } 
  $p0 = 0.0;

  $h *= $this->{L};

  return $h; 
}

sub aPotential
{
  my $a0 = 0.0;
  my $ax = 0.0;
  my $ay = 0.0;
  my $as = 0.0;
  return ($a0, $ax, $ay, $as);
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
