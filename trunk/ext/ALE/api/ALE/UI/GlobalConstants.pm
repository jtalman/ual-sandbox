package ALE::UI::GlobalConstants;

use strict;
use Carp;

# This is the contaner for global physical constants
# Copies of these constant are stored in PacBeamDef.h

sub new
{
  my $type = shift;

  my $this = {};

  $this->{"mass"}   = 0.9382796;
  $this->{"clight"} = 2.99792458e+8 ;
  $this->{"charge"} = 1.;

  return bless $this, $type;
}

sub getMass
{

  my $this   = shift;
  my $mass = $this->{"mass"};
  return $mass;
}

sub getClight
{

  my $this   = shift;
  my $clight =  $this->{"clight"};
  return $clight;
}

sub getCharge
{

  my $this   = shift;
  my $charge = $this->{"charge"};
  return $charge;
}

1;