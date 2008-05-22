package UALUSR::Shell;

use strict;
use Carp;
use vars qw(@ISA);

use lib ("$ENV{UAL_EXTRA}/ALE/api");
use ALE::UI::Shell;
@ISA = qw(ALE::UI::Shell);

sub new
{
  my $type = shift;
  my %params = @_;
  my $this = new ALE::UI::Shell(%params);

  return bless $this, $type;
}

# Add misalignments

sub addMisalignment
{
  my $this   = shift;
  my %params = @_; 

  # Translate input arguments

  my $pattern =  " "; if(defined $params{"elements"})
			{$pattern  = $params{"elements"}; }

  my $arg_counter = 0;
  my $sigx     = 0; if(defined $params{"dx"})  
			{$sigx     = $params{"dx"};   $arg_counter++;}
  my $sigy     = 0; if(defined $params{"dy"})  
			{$sigy     = $params{"dy"};   $arg_counter++;}
  my $sigtheta = 0; if(defined $params{"tilt"})
			{$sigtheta = $params{"tilt"}; $arg_counter++;}

  my $engine =   0;  if(defined $params{"engine"})
			{$engine   =   $params{"engine"};}
  my $cut    =   3;  if(defined $params{"cut"})   
			{$cut      =   $params{"cut"};} 

  # Set up smf keys
  my ($irKey, $dxKey, $dyKey, $tiltKey) = (0, 0, 0, 0); 

  my $smfMap = $this->{"shell"}->map();

  $smfMap->attribKeyFromString("n",     \$irKey);
  $smfMap->attribKeyFromString("dx",    \$dxKey);
  $smfMap->attribKeyFromString("dy",    \$dyKey);
  $smfMap->attribKeyFromString("tilt",  \$tiltKey);

  # Select elements
  my $lattice = $this->{"lattice"};
  my @elemIndices =  $lattice->indexes($pattern);

  my ($i, $j, $jk);

  my $rvalue = 1.0;
  my $element = 0;

  # - USR extension --------------------------------
  my @delx;
  my @dely;
  my @deltheta;
  # --------------------------------------------------

  for($i=0; $i < $#elemIndices + 1; $i++){
    $element = $lattice->element($elemIndices[$i]); 

    if($sigx){
    	if($engine) { $rvalue = $engine->getran($cut);}
        $element->add(($sigx*$rvalue)*$dxKey);
        ${\@delx}[$i] = $sigx*$rvalue; # USR extension
    }

    if($sigy){
    	if($engine) { $rvalue = $engine->getran($cut);}
        $element->add(($sigy*$rvalue)*$dyKey);
        ${\@dely}[$i] = $sigx*$rvalue; # USR extension
    }

    if($sigtheta){
    	if($engine) { $rvalue = $engine->getran($cut);}
        $element->add(($sigtheta*$rvalue)*$tiltKey);
        ${\@deltheta}[$i] = $sigtheta*$rvalue; # USR extension
    }
 
    if($engine) {
       my $ir = $element->get($irKey);
       my $fake_counter = (4.*$ir - 1.)*$arg_counter;
       for($jk = 0; $jk < $fake_counter; $jk++){
		$rvalue = $engine->getran($cut);
       }
    }
  }
  return (\@elemIndices,\@delx,\@dely,\@deltheta); # USR extension
}


# New method for getting erect magnet strengths; 

sub getErectMagnetStrengths
{
  my $this   = shift;
  my %params = @_; 

  # Translate input arguments

  my $pattern =  " "; if(defined $params{"elements"})
			{$pattern  = $params{"elements"}; }
  my $multindex = 0; if(defined $params{"multindex"})
			{$multindex = $params{"multindex"}; }

  # Set up smf keys
  my ($klKey, $multKey) = (0, 0);

  my $smfMap = $this->{"shell"}->map();

  $smfMap->attribKeyFromString("kl",  \$klKey);
  $smfMap->bucketKeyFromString("mult",  \$multKey);

  # Select elements
  my $lattice = $this->{"lattice"};
  my @elemIndices = $lattice->indexes($pattern);

  my $i;
  my $kl = $multKey->attribKey($klKey->index, $multindex);

  my $rvalue = 1.0;
  my $element = 0;
  my @klvalue = [];

  for($i=0; $i < $#elemIndices + 1; $i++){
    $element = $lattice->element($elemIndices[$i]);
    $klvalue[$i] = $element->get($kl);
  }
  return (\@elemIndices,\@klvalue);
}

# New method for getting skew  magnet strengths; 

sub getSkewMagnetStrengths
{
  my $this   = shift;
  my %params = @_; 

  # Translate input arguments

  my $pattern =  " "; if(defined $params{"elements"})
			{$pattern  = $params{"elements"}; }
  my $multindex = 0; if(defined $params{"multindex"})
			{$multindex = $params{"multindex"}; }

  # Set up smf keys
  my ($ktlKey, $multKey) = (0, 0);

  my $smfMap = $this->{"shell"}->map();

  $smfMap->attribKeyFromString("ktl",  \$ktlKey);
  $smfMap->bucketKeyFromString("mult",  \$multKey);

  # Select elements
  my $lattice = $this->{"lattice"};
  my @elemIndices =  $lattice->indexes($pattern);

  my $i;
  my $ktl = $multKey->attribKey($ktlKey->index, $multindex);

  my $rvalue = 1.0;
  my $element = 0;
  my @ktlvalue = [];

  for($i=0; $i < $#elemIndices + 1; $i++){
    $element = $lattice->element($elemIndices[$i]);
    $ktlvalue[$i] = $element->get($ktl);
  }
  return (\@elemIndices,\@ktlvalue);
}

1;

