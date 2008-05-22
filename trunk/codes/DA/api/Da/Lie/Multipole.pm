package Da::Lie::Multipole;

use strict;
use Carp;
use vars qw(@ISA);

use Da::Const qw($X_ $PX_ $Y_ $PY_ $CT_ $DE_);
use Da::Lie::Integrator;

@ISA = qw(Da::Lie::Integrator);

my $k  = [];
my $kt = [];
my $k_kt_size = 0;

sub new
{
  my $type   = shift;
  my %params = @_;

  my $this = Da::Lie::Integrator->new(@_);

  $this->{KL}  = $params{KL};
  $this->{KTL} = $params{KTL};

  return bless $this, $type;  
}

sub propagate
{
  my ($this, $object, $beam_att) = @_;
  
  $this->initialize();
  $this->SUPER::propagate($object, $beam_att);
  $this->erase();
}

sub aPotential
{
  my($self, $p) = @_;

  my ($a0, $ax, $ay, $as) = $self->SUPER::aPotential($p);

  my $i;
  my $tmp = 0.0;
  my $asi = 0.0;
  for($i = $k_kt_size; $i >= 0; $i--){
     $as *= $p->value($X_);
     $as -= $p->value($Y_)*$asi;

     $asi *= $p->value($X_);
     $asi += $p->value($Y_)*$tmp;

     if($i > 0){     
       $as  -= $k->[$i-1]/$i;
       $asi -= $kt->[$i-1]/$i;
     }     

     $tmp  = $as + 0.0;	
  }
  
  return ($a0, $ax, $ay, $as);  

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
  if($l <= 0.0) { croak "Da::Lie::Multipole: L, length, <= 0 \n"; }

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
