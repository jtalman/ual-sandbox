package HardEdge;

use strict;
use Carp;
use vars qw(@ISA);

use Da::Const qw($X_ $PX_ $Y_ $PY_ $CT_ $DE_);
use Da::Lie::Integrator;

@ISA = qw(Da::Lie::Integrator);

sub new
{
  my $type   = shift;
  my %params = @_;

  my $self = new Da::Lie::Integrator(@_);

  $self->{K}     = 0.0   unless defined ( $self->{K}    = $params{K} );

  return bless $self, $type;  
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

  # Hamiltonian

  my $x2 = $p0->value($X_)*$p0->value($X_);
  my $y2 = $p0->value($Y_)*$p0->value($Y_);
  
  $h  = 3.*$x2*$p0->value($Y_)*$p0->value($PY_);
  $h -= 3.*$y2*$p0->value($X_)*$p0->value($PX_);  
  $h +=    $y2*$p0->value($Y_)*$p0->value($PY_); 
  $h -=    $x2*$p0->value($X_)*$p0->value($PX_); 
  
  $h *= $this->{K}/12.;  

  return $h; 
}

1; 
