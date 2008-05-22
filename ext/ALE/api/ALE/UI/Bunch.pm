package ALE::UI::Bunch;

use strict;
use Carp;

use lib  ("$ENV{UAL_PAC}/api");

sub new
{
  my $type = shift;
  my $size = shift;
  my $this = {};
  $this->{"bunch"} = new Pac::Bunch($size);  
  # $this->{"mass"} = 0.93827231;
  # $this->{"energy"} = 1.e+12;
  # $this->{"bunch"}->mass($this->{"mass"});
  # $this->{"bunch"}->energy($this->{"energy"});
  return bless $this, $type;
}

sub setBeamAttributes
{

   my ($this, $energy, $mass) = @_;
   # $this->{"mass"} = $mass;
   # $this->{"energy"} = $energy;
   $this->{"bunch"}->mass($mass);
   $this->{"bunch"}->energy($energy);	
}

sub setPosition
{
   my ($this, $i, $x, $px, $y, $py, $ct, $dp) = @_;
   my $p = new Pac::Position();
   $p->set($x, $px, $y, $py, $ct, $this->_dp2de($dp));
   $this->{"bunch"}->position($i, $p);
}

sub getPosition
{
   my ($this, $i) = @_;
   my $p = $this->{"bunch"}->position($i);
   return ($p->x, $p->px, $p->y, $p->py, $p->ct, $this->_de2dp($p->de)); 
}

sub getSize(){
  my $this = shift;
  return $this->{"bunch"}->size;
}


sub getPacBunch
{
  my $this = shift;
  return $this->{"bunch"};
}

sub _dp2de
{
  my ($this, $dp) = @_;

  my ($e0, $p0, $m0, $p, $e);

  $e0 = $this->{"bunch"}->energy();
  $m0 = $this->{"bunch"}->mass();

  $p0 = $e0*$e0 - $m0*$m0;
  $p0 = sqrt($p0);

  $p = $p0*(1.0 + $dp);
  $e = sqrt($p*$p + $m0*$m0);

  return ($e - $e0)/$p0;
}

sub _de2dp {

    my ($this, $de) = @_;

    my ($e0, $p0, $m0, $p, $e);

    $e0 = $this->{"bunch"}->energy();
    $m0 = $this->{"bunch"}->mass();

    $p0 = $e0*$e0 - $m0*$m0;
    $p0 = sqrt($p0);

    $e = $p0*$de + $e0;
    $p = sqrt(($e - $m0)*($e + $m0));
    return ($p - $p0)/$p0;
}

1;
