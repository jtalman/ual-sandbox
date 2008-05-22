package Da::Rk::Integrator;

use strict;
use Carp;
use vars qw(@ISA);

use Da::Const qw($X_ $PX_ $Y_ $PY_ $CT_ $DE_);
use Zlib::Tps;

use Da::Integrator;
@ISA = qw(Da::Integrator);

sub new
{
  my $type   = shift;
  my %params = @_;

  my $this = new Da::Integrator(@_);

  $this->{L} = 0.0 unless defined ( $this->{L} = $params{L} );
  $this->{N} = 1   unless defined ( $this->{N} = $params{N} );

  return bless $this, $type;
}

sub transform
{
  my ($this,  $p) = @_;
  my $x = 0.0;

  my $nsteps = $this->{N};
  if($nsteps <= 0) { croak "Da::Rk::Integrator: N, number of steps, <= 0 \n"; }

  my $h = $this->{L}/$nsteps;
  if(($x + $h) == $x) { croak "Da::Rk::Integrator: L/N, step size, is too small \n"; }

# Take nsteps steps

  my $k;
  my $pv = [];

  for($k=0; $k < @$p; $k++) {
	push @$pv,  0.0; 
  }
  
  for($k=0; $k < $nsteps; $k++){
    $this->rk4($p, $pv, $x, $h);
    $x += $h;
  }

}

sub rk4
{

  my ($this, $p, $pv, $x, $h) = @_;

  my $size = @$p;

  my $pt   = [];
  my $dpt  = [];
  my $dpm  = [];

  my $i;
  for($i=0; $i < $size; $i++) { 
     push @$pt,  0.0;
     push @$dpt, 0.0;
     push @$dpm, 0.0; 
  }
 
  my $hh = $h*0.5;
  my $xh = $x + $hh;

  # First step

  $this->rightSide($p, $pv, $x);
  for($i=0; $i < $size; $i++) { 
    $pt->[$i] = $p->[$i] + $hh*$pv->[$i]; 
  }

  # Second step

  $this->rightSide($pt, $dpt, $xh);
  for($i=0; $i < $size; $i++) {
    $pt->[$i] = $p->[$i] + $hh*$dpt->[$i];
  }

  # Third step

  $this->rightSide($pt, $dpm, $xh);        
  for($i=0; $i < $size; $i++){
    $pt->[$i]   = $p->[$i] + $h*$dpm->[$i];
    $dpm->[$i] += $dpt->[$i];
  }

  # Fourth step

  $this->rightSide($pt, $dpt, $x+$h); 
  for($i=0; $i < $size; $i++){
    $p->[$i] += ($h/6.0)*($pv->[$i] + $dpt->[$i] + 2.0*$dpm->[$i]);
  }

}

sub rightSide
{
   my ($this, $p, $pv, $x) = @_;

   my $size = @$p;
   if($size < $PX_) { return; }

   # Beam

   my $v0byc  = $this->{v0byc}; 
   my $charge = $this->{charge};

   # Field

   my ($b0, $bx, $by, $bz) = $this->bField($x, $p);

   $bx *= $charge;
   $by *= $charge;
   $bz *= $charge;

   # ps/p0

   my $psp0 = $this->psp0($p);	

   # 1 + x/R

   my $dxR       = (1. + $b0*$p->[$X_]);
   my $dxR_by_ps = $dxR/$psp0;

   # 

   $pv->[$X_]   = $dxR_by_ps*$p->[$PX_];
   $pv->[$PX_]  = $b0*$psp0;
   $pv->[$PX_] -= $by*$dxR; 

   if($size > $PY_) {   
       $pv->[$Y_]   = $dxR_by_ps*$p->[$PY_];
       $pv->[$PY_]  = $bx*$dxR;

       $pv->[$PX_] += $bz*$pv->[$Y_];
       $pv->[$PY_] -= $bz*$pv->[$X_];
   }
   if($size > $DE_) {
       $pv->[$CT_]  = $p->[$DE_] + 1./$v0byc;
       $pv->[$CT_] *= $dxR_by_ps;
       $pv->[$CT_] *= -1.0;
       $pv->[$CT_] += 1./$v0byc;
       $pv->[$DE_] *= 0.0;
   }

}


sub bField
{
  my($self, $x, $p) = @_;

  my $b0 = 0.0;
  my $bx = 0.0;
  my $by = 0.0;
  my $bz = 0.0;

  return ($b0, $bx, $by, $bz);

}

1;

