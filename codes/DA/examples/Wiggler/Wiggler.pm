package Wiggler;

use strict;
use Carp;
use vars qw(@ISA);

use Da::Const qw($X_ $PX_ $Y_ $PY_ $CT_ $DE_);
use Da::Rk::Integrator;

@ISA = qw(Da::Rk::Integrator);

sub new
{
  my $type   = shift;
  my %params = @_;

  my $self = new Da::Rk::Integrator(@_);

  $self->{B0}     = 0.0   unless defined ( $self->{B0}    = $params{B0} );
  $self->{KX}     = 0.0   unless defined ( $self->{KX}    = $params{KX} );
  $self->{PHASE}  = 0.0   unless defined ( $self->{PHASE} = $params{PHASE} );

  $self->{KZ} = 8*atan2(1,1)/$self->{L} unless defined ( $self->{KZ} = $params{KZ} );

  $self->{SH_NUMBER}  = 5.0 unless defined ( $self->{SH_NUMBER} = $params{SH_NUMBER} );

  return bless $self, $type;  
}


sub bField
{
  my($self, $s, $p) = @_;
  my ($b0, $kx, $ky, $kz, $phase) = $self->attributes();

  my $xk = $p->[$X_]*$kx;
  my $yk = $p->[$Y_]*$ky;

  my ($one_by_R, $bx, $by, $bz) = $self->SUPER::bField($s, $p);

  $bx  = $self->lsh($yk) + 0.0;
  $by  = $self->lcn($xk) + 0.0;
  $bz  = $bx*$by;

  $bx *= $self->lsn($xk);
  $bx *= -cos($kz*$s + $phase)*$b0*$kx/$ky;

  $by *= $self->lch($yk);
  $by *= cos($kz*$s + $phase)*$b0;

  $bz *= -sin($kz*$s + $phase)*$b0*$kz/$ky;

  return ($one_by_R, $bx, $by, $bz);
}

sub attributes
{
  my $self = shift;

  my $b0    = $self->{B0};
  my $kx    = $self->{KX};
  my $kz    = $self->{KZ};
  my $ky    = sqrt($kx*$kx + $kz*$kz);
  my $phase = $self->{PHASE};
  
  return ($b0, $kx, $ky, $kz, $phase);
}

sub lsh
{
  my ($self, $x) = @_;

  my $term = 1.0*$x;
  my $sum  = 0.0 + $term;

  my $i;
  my $now = $self->{SH_NUMBER};
  for($i=2; $i <= $now; $i++){
	$term *= $x;
        $term /= $i;
        if($i%2 != 0) { $sum  += $term; }
  }

  return $sum;
}

sub lsn
{
  my ($self, $x) = @_;

  my $term = 1.0*$x;
  my $sum  = 0.0 + $term;

  my $i;
  my $now = $self->{SH_NUMBER};
  for($i=2; $i <= $now; $i++){
	$term *= $x;
        $term /= -$i;
        if($i%2 != 0) {
		$term *= -1.;
  		$sum  += $term;
	}
  }

  return $sum;
}

sub lch
{
  my ($self, $x) = @_;
 
  my $term = 1.0*$x;
  my $sum  = 1.0 + 0.0*$term;  

  my $i;
  my $now = $self->{SH_NUMBER};
  for($i = 2; $i <= $now; $i++){
	$term *= $x;
	$term /= $i;
	if($i%2 == 0) { $sum  += $term; }
  }
  return $sum;
}


sub lcn
{
  my ($self, $x) = @_;
 
  my $term = (-1.0)*$x;
  my $sum  = 1.0 + 0.0*$term;  

  my $i;
  my $now = $self->{SH_NUMBER};
  for($i = 2; $i <= $now; $i++){
	$term *= $x;
	$term /= -$i;
	if($i%2 == 0) {
 		$term *= -1.;
		$sum  += $term;
 	}
  }
  return $sum;
}

1; 
