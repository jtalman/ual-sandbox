package Teapot::Main;

use lib ("$ENV{UAL_TEAPOT}/api/blib/$ENV{UAL_ARCH}/arch", 
         "$ENV{UAL_TEAPOT}/api/blib/$ENV{UAL_ARCH}/lib");

use strict;
use Carp;
use vars qw($VERSION @ISA @EXPORT $AUTOLOAD);

require Exporter;
require DynaLoader;
require AutoLoader;

@ISA = qw(Exporter DynaLoader);
@EXPORT = qw();
$VERSION = '1.0';

bootstrap Teapot::Main $VERSION;

package Teapot::Main;

sub hsteer
{
  my $this = shift;
  my @input = $this->read_steer(@_);
  $this->steer_(@input, 'h');
}

sub vsteer
{
  my $this = shift;
  my @input = $this->read_steer(@_);
  $this->steer_(@input, 'v');
}

sub ftsteer
{
  my $this = shift;
  my @input = $this->read_ftsteer(@_);
  $this->ftsteer_(@input);
}

sub tunethin
{
  my $this = shift;
  my @input = $this->read_fit(@_);
  $this->tunethin_(@input);
}

sub chromfit
{
  my $this = shift;
  my @input = $this->read_chromfit(@_);
  $this->chromfit_(@input);
}

sub decouple
{
  my $this = shift;
  my @input = $this->read_decouple(@_);
  $this->decouple_(@input);
}

sub indices
{
   my ($self, $regex) = @_;

   my (@v, $i);
   for($i = 0; $i < $self->size(); $i++){
	if($self->element($i)->genName =~ $regex) { push @v, $i;}
   }
   return @v;   
}

sub read_steer
{
   my $this = shift;
   my %params = @_;

   my $beam = $params{beam}; 
   if(not defined $beam) { croak "read_steer : beam is not defined \n"; }

   my $orbit = new Pac::Position();
   my $delta = $params{delta}; 
   if(defined $delta) { $orbit->de($delta); }

   my $adjusters = $params{adjusters}; 
   if(not defined $adjusters) { croak "read_steer : adjusters are not defined \n"; }
   my @ads = $this->indices($adjusters);

   my $detectors = $params{detectors};
   if(not defined $detectors) { croak "read_steer : detectors are not defined \n"; } 
   my @dets = $this->indices($detectors);

   my $methodName = $params{method};
   my $method;
   if($methodName == "SB") { $method = 1;}
   else {$method = 0;}  
   if(not defined $methodName) {$method = 0;} 

   return ($orbit, $beam, \@ads, \@dets, $method);  
}

sub read_ftsteer
{
   my $this = shift;
   my %params = @_;

   my $beam = $params{beam}; 
   if(not defined $beam) { croak "ftread_steer : beam is not defined \n"; }

   my $orbit = $params{orbit};
   if(not defined $orbit){ $orbit = new Pac::Position();}

   my $delta = $params{delta}; 
   if(defined $delta) { $orbit->de($delta); }

   my $hadjusters = $params{hadjusters}; 
   if(not defined $hadjusters) { croak "ftread_steer : hadjusters are not defined \n"; }
   my @hads = $this->indices($hadjusters);

   my $vadjusters = $params{vadjusters}; 
   if(not defined $vadjusters) { croak "ftread_steer : vadjusters are not defined \n"; }
   my @vads = $this->indices($vadjusters);

   my $hdetectors = $params{hdetectors};
   if(not defined $hdetectors) { croak "ftread_steer : hdetectors are not defined \n"; } 
   my @hdets = $this->indices($hdetectors);

   my $vdetectors = $params{vdetectors};
   if(not defined $vdetectors) { croak "ftread_steer : vdetectors are not defined \n"; } 
   my @vdets = $this->indices($vdetectors);

   my $maxdev = $params{maxdev}; 
   if(not defined $maxdev) {$maxdev = 0.01;} 

   my $tw = $params{tw};
   if(not defined $tw) { croak "ftread_steer : tw are not defined \n"; } 

   my $methodName = $params{method};
   my $method;
   if($methodName == "SB") { $method = 1;}
   else {$method = 0;}  
   if(not defined $methodName) {$method = 0;} 

   return ($orbit, $beam, \@hads, \@hdets, \@vads, \@vdets, $maxdev, $tw, $method);  
}

sub read_fit
{
   my $this = shift;
   my %params = @_;

   my $beam = $params{beam}; 
   if(not defined $beam) { croak "read_fit : beam is not defined \n"; }

   my $orbit = new Pac::Position();
   my $delta = $params{delta}; 
   if(defined $delta) { $orbit->de($delta); }

   my $bf = $params{bf}; 
   if(not defined $bf) { croak "read_fit : bf is not defined \n"; }
   my @af = $this->indices($bf);

   my $bd = $params{bd};
   if(not defined $bd) { croak "read_fit : bd is not defined \n"; } 
   my @ad = $this->indices($bd);

   my $mux = $params{mux}; 
   if(not defined $mux) { croak "read_fit : mux is not defined \n"; }

   my $muy = $params{muy};
   if(not defined $muy) { croak "read_fit : muy is not defined \n"; } 

   my $method = $params{method}; 
   if(not defined $method) { $method = '*'; }

   my $numtries = $params{numtries}; 
   if(not defined $numtries) { $numtries = 100; } 
  
   my $tolerance = $params{tolerance}; 
   if(not defined $tolerance) { $tolerance = 1.e-6; } 

   my $stepsize = $params{stepsize}; 
   if(not defined $stepsize) { $stepsize = 0.0; }  

   return ($beam, $orbit, \@af, \@ad, $mux, $muy, $method, $numtries, $tolerance, $stepsize);  
}

sub read_chromfit
{
   my $this = shift;
   my %params = @_;

   my $beam = $params{beam}; 
   if(not defined $beam) { croak "read_chromfit : beam is not defined \n"; }

   my $orbit = new Pac::Position();
   my $delta = $params{delta}; 
   if(defined $delta) { $orbit->de($delta); }

   my $bf = $params{bf}; 
   if(not defined $bf) { croak "read_chromfit : bf is not defined \n"; }
   my @af = $this->indices($bf);

   my $bd = $params{bd};
   if(not defined $bd) { croak "read_chromfit : bd is not defined \n"; } 
   my @ad = $this->indices($bd);

   my $mux = $params{chromx}; 
   if(not defined $mux) { croak "read_chromfit : chromx is not defined \n"; }

   my $muy = $params{chromy};
   if(not defined $muy) { croak "read_chromfit : chromy is not defined \n"; } 

   my $method = $params{method}; 
   if(not defined $method) { $method = '*'; }

   my $numtries = $params{numtries}; 
   if(not defined $numtries) { $numtries = 10; } 
  
   my $tolerance = $params{tolerance}; 
   if(not defined $tolerance) { $tolerance = 1.e-4; } 

   my $stepsize = $params{stepsize}; 
   if(not defined $stepsize) { $stepsize = 0.0; }  

   return ($beam, $orbit, \@af, \@ad, $mux, $muy, $method, $numtries, $tolerance, $stepsize);  
}

sub read_decouple
{
   my $this = shift;
   my %params = @_;

   my $beam = $params{beam}; 
   if(not defined $beam) { croak "read_decouple : beam is not defined \n"; }

   my $orbit = new Pac::Position();
   my $delta = $params{delta}; 
   if(defined $delta) { $orbit->de($delta); }

   my $a11 = $params{a11}; 
   if(not defined $a11) { croak "read_decouple : a11 is not defined \n"; }
   my @aa11 = $this->indices($a11);

   my $a12 = $params{a12};
   if(not defined $a12) { croak "read_decouple : a12 is not defined \n"; } 
   my @aa12 = $this->indices($a12);

   my $a13 = $params{a13}; 
   if(not defined $a13) { croak "read_decouple : a13 is not defined \n"; }
   my @aa13= $this->indices($a13);

   my $a14 = $params{a14};
   if(not defined $a14) { croak "read_decouple : a14 is not defined \n"; } 
   my @aa14 = $this->indices($a14);

   my $bf = $params{bf}; 
   if(not defined $bf) { croak "read_decouple : bf is not defined \n"; }
   my @abf = $this->indices($bf);

   my $bd = $params{bd};
   if(not defined $bd) { croak "read_decouple : bd is not defined \n"; } 
   my @abd = $this->indices($bd);

   my $mux = $params{mux}; 
   if(not defined $mux) { croak "read_decouple : mux is not defined \n"; }

   my $muy = $params{muy};
   if(not defined $muy) { croak "read_decouple : muy is not defined \n"; } 

   return ($beam, $orbit, \@aa11, \@aa12, \@aa13, \@aa14, \@abf, \@abd, $mux, $muy);  
}

package Teapot::Main;

1;
__END__
