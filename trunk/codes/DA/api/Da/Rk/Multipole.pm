package Da::Rk::Multipole;

use strict;
use Carp;
use vars qw(@ISA $k $kt);

use Da::Const qw($X_ $PX_ $Y_ $PY_ $CT_ $DE_);
use Da::Rk::Integrator;

@ISA = qw(Da::Rk::Integrator);

$k  = [];
$kt = [];
my $k_kt_size = 0;

sub new
{
  my $type   = shift;
  my %params = @_;

  my $this = new Da::Rk::Integrator(@_);

  $this->{KL}  = $params{KL};
  $this->{KTL} = $params{KTL};

  return bless $this, $type;  
}

sub transform
{
  my ($this, $object) = @_;
  
  $this->initialize();
  $this->SUPER::transform($object);
  $this->erase();
}

sub bField
{
  my($self, $s, $p) = @_;

  my ($b0, $bx, $by, $bz) = $self->SUPER::bField($s, $p);

  my $i;
  my $tmp = 0.0;
  for($i = $k_kt_size - 1; $i >= 0; $i--){
     $by *= $p->[$X_];
     $by -= $p->[$Y_]*$bx;
     $by += $k->[$i];

     $bx *= $p->[$X_];
     $bx += $p->[$Y_]*$tmp;
     $bx += $kt->[$i];     

     $tmp  = $by + 0.0;	
  }

  return ($b0, $bx, $by, $bz);
}

sub initialize
{
  my $this = shift;

  $this->erase; 

  my ($tmp_k, $tmp_kt);
  $tmp_k  = [] unless defined($tmp_k  = $this->{KL}); 
  $tmp_kt = [] unless defined($tmp_kt = $this->{KTL}); 

  my $ksize  = @$tmp_k;
  my $ktsize = @$tmp_kt;

  my $l  = $this->{L};
  if($l <= 0.0) { croak "Da::Rk::Multipole: L, length, <= 0 \n"; }

  my ($i);
  for($i = 0; $i < $ksize;  $i++) { push @$k,  $tmp_k->[$i]/$l; }
  for($i = 0; $i < $ktsize; $i++) { push @$kt, $tmp_kt->[$i]/$l; }

  if($ksize > $ktsize) { 
     $k_kt_size  = $ksize;
     for($i = $ktsize; $i < $k_kt_size; $i++) { push @$kt, 0.0;}
  }
  else{ 
     $k_kt_size  = $ktsize; 
     for($i = $ksize;  $i < $k_kt_size; $i++) { push @$k,  0.0;}
  }

}

sub erase
{
  $k  = [];
  $kt = [];
  $k_kt_size = 0;
}

1; 
