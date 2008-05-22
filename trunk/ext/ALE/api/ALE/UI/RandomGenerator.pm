package ALE::UI::RandomGenerator;

use strict;
use Carp;
use vars qw(@ISA);

use lib ("$ENV{UAL_TEAPOT}/api");
use Teapot::Math;
# @ISA = qw(Teapot::RandomGenerator);

sub new
{
  my $type = shift;
  my $iseed = shift;

  my $this = {};
  $this->{adaptee} = new Teapot::RandomGenerator($iseed);

  return bless $this, $type;
}

sub getSeed
{
  my $this = shift;
  return $this->{adaptee}->getSeed();
}

sub setSeed
{
  my $this = shift;
  my $iseed = shift;
  $this->{adaptee}->setSeed($iseed);
}

sub getran
{
  my $this = shift;
  my $cut = shift;
  return $this->{adaptee}->getran($cut);
}

1;
