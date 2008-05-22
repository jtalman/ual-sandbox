package Accsim::Facade;

use strict;
use Carp;

use lib  ("$ENV{UAL_PAC}/api", "$ENV{UAL_ACCSIM}/api", "$ENV{PGPLOT}/api");
use Pac; 
use Accsim::Bunch;
use Accsim::Plot;
use PGPLOT;

sub new
{
  my $type = shift;
  my $this = {}; 
  return bless $this, $type; 
}

# Inject particles

sub inject
{
   my ($this, $bunch, $injCenter, $twiss, $emittance, $cut, $ref_seed) = @_; 

   my $seed = $$ref_seed;

   my $bunchGenerator = new Accsim::BunchGenerator;
   $bunchGenerator->shift($bunch, $injCenter);

   # Make transverse distributions 
   # Default : ACCSIM idist<xy> = 3 : binomial, m = 3

   my $bunchType = 3;
   my $emitXY = new Pac::Position();
   $emitXY->x($emittance->x());
   $emitXY->y($emittance->y());
   $bunchGenerator->addBinomialEllipses(
	$bunch, $bunchType, $twiss, $emitXY, $seed);

   # Make longitudinal distribution 
   # Default: ACCSIM idistl = 4 : uniform in phase,Gaussian in energy

   my $phaseHalfWidth = new Pac::Position();
   $phaseHalfWidth->ct($emittance->ct());
   $bunchGenerator->addUniformRectangles($bunch, $phaseHalfWidth, $seed); 

   my $energyRMS = new  Pac::Position();  
   $energyRMS->de($emittance->de());
   $bunchGenerator->addGaussianRectangles($bunch, $energyRMS, $cut, $seed);

   $$ref_seed = $seed;
}

# Estimate the beam emittance

sub emit 
{
   my ($this, $bunch) = @_;  

   my $orbit = new Pac::Position;
   my $twiss = new Pac::TwissData;
   my $rms   = new Pac::Position;

   print "\nAccsim emit command.\n";
   print "----------------------\n";

   my $bunchAnalyzer  = new Accsim::BunchAnalyzer; 
   $bunchAnalyzer->getRMS($bunch, $orbit, $twiss, $rms);

   print "\nOrbit : \n"; 
   print sprintf(" % -12.5e  % -12.5e % -12.5e % -12.5e % -12.5e % -12.5e\n",
		  $orbit->x(),  $orbit->px(), 
		  $orbit->y(),  $orbit->py(), 
		  $orbit->ct(), $orbit->de());

   print "\nRMS (x-px [m-rad], y-py [m-rad], ct [m], de/p []) : \n";   
   print sprintf(" % -12.5e  % -12.5e % -12.5e % -12.5e\n",
		  $rms->x(), $rms->y(), $rms->ct(), $rms->de());

   print "\nTwiss parameters (betaX, alphaX, betaY, alphaY) : \n";     
   print sprintf(" % -12.5e  % -12.5e % -12.5e  % -12.5e\n",
		 $twiss->beta(0), $twiss->alpha(0), 
		 $twiss->beta(1), $twiss->alpha(1));

}

# Produces scatterplots and histograms of particle positions.

sub scat
{
    my $this   = shift;
    my %params = @_;
    my $bunch  = $params{"bunch"};
    my $device = $params{"device"};
    my $width  = 4.0;   if( defined $params{"width"}  ) { $width   = $params{"width"}; } 
    my $ratio  = 0.8;   if( defined $params{"ratio"}  ) { $ratio   = $params{"ratio"}; }
    my $bgred   = 0.8 ; if( defined $params{"bgred"}  ) { $bgred   = $params{"bgred"}; }
    my $bggreen = 0.8 ; if( defined $params{"bggreen"}) { $bggreen = $params{"bggreen"};}
    my $bgblue  = 0.7 ; if( defined $params{"bgblue"} ) { $bgblue  = $params{"bgblue"};}
    my $text    = ""  ; if( defined $params{"text"}   ) { $text    = $params{"text"};}

    my $plotter = new Accsim::Plot();
    $plotter->scat($bunch, $device, $width, $ratio, $bgred, $bggreen, $bgblue, $text);
}

1;
