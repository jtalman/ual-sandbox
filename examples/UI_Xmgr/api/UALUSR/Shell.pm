package UALUSR::Shell;

use strict;
use Carp;
use vars qw(@ISA);

use lib ("$ENV{UAL_EXTRA}/ALE/api");
use ALE::UI::Shell;
@ISA = qw(ALE::UI::Shell);

use UALUSR::SimpleTwissp;
use UALUSR::SimpleTrackingp;

sub run
{
  my $this   = shift;
  my %params = @_;
  my $shell_there = $this;

  my ($bunch, $file, $turns, $step) = (0, "./fort.8", 1, 1);
  my $orbit = new Pac::Position;

  my $message;
  $message = "\nrun \n";   
  $shell_there->_printLogMessage($message);

  if(defined $params{"bunch"})   { $bunch = $params{"bunch"}; }
  if(defined $params{"print"})   { $file  = $params{"print"}; }
  if(defined $params{"turns"})   { $turns = $params{"turns"}; }
  if(defined $params{"step"})    { $step  = $params{"step"}; }
  if(defined $params{"orbit"})   { $orbit->set(@{$params{"orbit"}}); }
 
  $message = "start = " . time . "\n";
  $shell_there->_printLogMessage($message);  

  my $service = new ALE::UI::SimpleTrackingp();
  $service->run($file, $turns, $step, $this, $bunch->{"bunch"}, $orbit);

  $message = "end   = " . time . "\n";
  $shell_there->_printLogMessage($message);  
 
}

1;

