package ALE::UI::Shell;

use strict;
use Carp;

use lib  ("$ENV{UAL_PAC}/api", "$ENV{UAL_ZLIB}/api", "$ENV{UAL_TEAPOT}/api", "$ENV{UAL_EXTRA}/PAC/api");

# UAL core libraries
use Pac; use Zlib::Tps; use Teapot;

# UAL extensions
use PAC::MAD::SMF; use PAC::FTPOT::Shell;

# Local classes
use ALE::UI::Bunch;
use ALE::UI::SimpleSurvey; 
use ALE::UI::SimpleTwiss;
use ALE::UI::SimpleTracking;
use ALE::UI::SimpleMatrix;
use ALE::UI::SimpleMapping;
use ALE::UI::RandomGenerator;
use ALE::UI::GlobalConstants;

my $DIM = 6;

sub new
{
  my $type = shift;

  my %params = @_;
  my $log =  "./log"; if(defined $params{"print"}) {$log  = $params{"print"}; }
  open(LOG, ">$log") || die "can't create $log";
  close(LOG);

  my $this = {};

  my $smf = new PAC::MAD::SMF();

  $this->{"shell"}   = new PAC::FTPOT::Shell($smf);
  $this->{"code"}    = new Teapot::Main();
  $this->{"lattice"} = 0;
  $this->{"beam"}    = new Pac::BeamAttributes();
  $this->{"orbit"}   = new Pac::Position();
  $this->{"bunch"}   = new Pac::Bunch(0);
  $this->{"space"}   = 0;
  $this->{"map"}     = 0;
  $this->{"twiss"}   = 0;
  $this->{"log"}     = $log;
  $this->{"constants"}     = new ALE::UI::GlobalConstants();

  return bless $this, $type;
}

# ************************************************************************
# Space of Taylor maps
# ************************************************************************

# Defines the space of Taylor maps

sub setMapAttributes
{

  my $this   = shift;
  my %params = @_;

  my $order = $params{"order"};


   my $service = new ALE::UI::SimpleMapping();
   $service->space($this, $order);

}

# ****************************************
# Accelerator design
# ****************************************

# Initialize the SMF from the MAD file

sub readMAD
{
  my $this   = shift;
  my %params = @_;

  my $file = $params{"file"};
  open (MAD,"<$file") or die "Can't find the MAD file $file \n"; 
  close(MAD);	

  my $id = 0; if(defined $params{"id"}) {$id  = $params{"id"}; }

  my $start = time;
  $this->{shell}->smf->restore("files" => [$file], "id" => $id);
  my $end   = time;

  my $message;
  $message = "\nreads the MAD input file " . $file . "\n";
  $message = $message . "start = " . $start . "\n";
  $message = $message . "end   = " . $end .  "\n";

  $this->_printLogMessage($message);
  
}

# Split elements

sub addSplit
{
  my $this = shift;

  my %params = @_;
  my $elements = $params{elements};
  my $ir       = $params{ir};

  $this->{shell}->split($elements, $ir);

}

# Define aperture parameters

sub addAperture
{
  my $this = shift;

  my %params = @_;
  my $elements = $params{elements};
  my $ap       = $params{aperture};

  my $smfMap = $this->{"shell"}->map();
  my ($shapeKey, $xsizeKey, $ysizeKey) = (0, 0, 0);

  my $shape = $ap->[0]; $smfMap->attribKeyFromString("shape", \$shapeKey);
  my $xsize = $ap->[1]; $smfMap->attribKeyFromString("xsize", \$xsizeKey);
  my $ysize = $ap->[2]; $smfMap->attribKeyFromString("ysize", \$ysizeKey); 

  # Get lattice
  my $lattice = $this->{"lattice"};
  
  if(defined $lattice && $lattice != 0) {

    # select lattice elements

    my @elemIndices =  $lattice->indexes($elements);

    my ($i, $element);
    for($i=0; $i < $#elemIndices + 1; $i++){
      $element = $lattice->element($elemIndices[$i]); 
      $element->add($shape*$shapeKey, 
		    $xsize*$xsizeKey, 
		    $ysize*$ysizeKey);
    } 
  } 
  else {

    # select design elements

    my $smf = $this->smf();

    my $it;
    for($it = $smf->elements->begin(); $it != $smf->elements->end(); $it++){
      if($it->second->name =~ /$elements/) { 
	$it->second->add($shape*$shapeKey, 
			 $xsize*$xsizeKey, 
			 $ysize*$ysizeKey); 
      }
    }  
  }

}

# Define elements as Taylor Maps

sub addMap
{
  my $this = shift;

  my %params = @_;
  my $elements = $params{elements};
  my $map_file = $params{map};

  my $map = new Zlib::VTps(6);
  $map->read($map_file);

  my $smf = $this->{shell}->smf;

  my $it;
  for($it = $smf->elements->begin(); $it != $smf->elements->end(); $it++){
    if($it->second->name =~ /$elements/) { 
      	  $it->second->map($map); 
    }
  }  

}

# Select a lattice

sub use
{
  my $this = shift;

  my %params = @_;
  my $line = $params{lattice};

  my $lattice;

  my $it  = $this->{shell}->smf->lattices->find($line);

  if($it != $this->{shell}->smf->lattices->end()) { $lattice = $it->second; }
  else { $lattice = $this->{shell}->lattice($line, $line) }

  $this->{code}->use($lattice);
  $this->{lattice} = $lattice;
}

# Write the SMF into the FTPOT file

sub writeFTPOT
{
   
  my $this = shift;
  my %params = @_;

  my $message;
  $message = "\nwrites the SMF data into the FTPOT file \n";
  $this->_printLogMessage($message);


   $message = "\n start           . " . time . "\n";
   $this->_printLogMessage($message);

  $this->{shell}->smf->store(%params);

   $message = "\n end             . " . time . "\n";
   $this->_printLogMessage($message);

}

# ****************************************
# Field errors, misalignments, etc.
# ****************************************

# Add element errors

sub addFieldError
{
  my $this   = shift;
  my %params = @_; 

  # Translate input arguments

  my $pattern =  " "; if(defined $params{"elements"}) {$pattern  = $params{"elements"}; }

  my $Rref =  1.; if(defined $params{"R"})  {$Rref  =  $params{"R"}; }
  my @dkl  =  (); if(defined $params{"b"}) {@dkl   =  @{$params{"b"}};}
  my @dktl  = (); if(defined $params{"a"}){@dktl  =  @{$params{"a"}};}

  my $engine = 0;  if(defined $params{"engine"}) {$engine  =   $params{"engine"};}
  my $cut    = 3;  if(defined $params{"cut"})   {$cut     =   $params{"cut"};} 

  # Set up smf keys
  my ($irKey, $angleKey, $klKey, $ktlKey, $multKey) = (0, 0, 0, 0, 0);

  my $smfMap = $this->{"shell"}->map();

  $smfMap->attribKeyFromString("n",     \$irKey);
  $smfMap->attribKeyFromString("kl",    \$klKey);
  $smfMap->attribKeyFromString("ktl",   \$ktlKey);
  $smfMap->attribKeyFromString("angle", \$angleKey);
  $smfMap->bucketKeyFromString("mult",  \$multKey);

  # Select elements
  my $lattice = $this->{"lattice"};
  my @elemIndices =  $lattice->indexes($pattern);

  my ($i, $j, $jk);
  my @kl;
  my @ktl;

  # Prepare a common part of coefficients: [1.e-4/(Rref)**n]
  for($i=0; $i < $#dkl + 1; $i++){
     for($j = 0; $j < $i; $j++){
   	$dkl[$i] /= $Rref;
     }
     $dkl[$i] *= 1.e-4;
     push @kl, $multKey->attribKey($klKey->index, $i);
  }
  for($i=0; $i < $#dktl + 1; $i++){
     for($j = 0; $j < $i; $j++){
   	$dktl[$i] /= $Rref;
     }
     $dktl[$i] *= 1.e-4;
     push @ktl, $multKey->attribKey($ktlKey->index, $i);
  }

  # Add errors : KLn(UAL) = [1.e-4/(Rref)**n]*ANGLE*KLn(table)
  # Bend : bzero = ANGLE
  # Quad : bzero = KL1*Rref;

  my $rvalue = 1.0;
  my ($element, $bzerol, $angle) = (0, 0, 0);
  for($i=0; $i < $#elemIndices + 1; $i++){
    $element = $lattice->element($elemIndices[$i]); 

    $angle = $element->get($angleKey);
    if($angle) { $bzerol  = $angle; }
    else       { $bzerol  = $element->get($kl[1])*$Rref; }

    for($j = 0; $j < $#dktl + 1; $j++){
        if($engine) { $rvalue = $engine->getran($cut);}
    	$element->add(($dktl[$j]*$bzerol*$rvalue)*$ktl[$j]);
    }

    for($j = 0; $j < $#dkl + 1; $j++){
        if($engine) { $rvalue = $engine->getran($cut);}
    	$element->add(($dkl[$j]*$bzerol*$rvalue)*$kl[$j]);
    }
    if($engine) {
       my $ir = $element->get($irKey);
       my $fake_counter = (4.*$ir - 1.)*($#dkl + 1 + $#dktl + 1);
       for($jk = 0; $jk < $fake_counter; $jk++){
		$rvalue = $engine->getran($cut);
       }
    }
  }
  
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
  for($i=0; $i < $#elemIndices + 1; $i++){
    $element = $lattice->element($elemIndices[$i]); 

    if($sigx){
    	if($engine) { $rvalue = $engine->getran($cut);}
        $element->add(($sigx*$rvalue)*$dxKey);
    }

    if($sigy){
    	if($engine) { $rvalue = $engine->getran($cut);}
        $element->add(($sigy*$rvalue)*$dyKey);
    }

    if($sigtheta){
    	if($engine) { $rvalue = $engine->getran($cut);}
        $element->add(($sigtheta*$rvalue)*$tiltKey);
    }
 
    if($engine) {
       my $ir = $element->get($irKey);
       my $fake_counter = (4.*$ir - 1.)*$arg_counter;
       for($jk = 0; $jk < $fake_counter; $jk++){
		$rvalue = $engine->getran($cut);
       }
    }
  }
  
}

# ****************************************
# Beam attributes
# ****************************************

sub setBeamAttributes
{
  my $this   = shift;
  my %params = @_;

  if(defined $params{energy} )   { $this->{beam}->energy($params{energy}); }
  if(defined $params{charge} )   { $this->{beam}->charge($params{charge}); }
  if(defined $params{mass} )     { $this->{beam}->mass($params{mass}); }
  if(defined $params{revfreq} )  { $this->{beam}->revfreq($params{revfreq}); }
  if(defined $params{macrosize} ){ $this->{beam}->macrosize($params{macrosize}); }

}

# ****************************************
# Analysis
# ****************************************

sub analysis
{

  my $this = shift;
  my $code = $this->{"code"};
  my $beam = $this->{"beam"};

  my $message;
  $message = "\nanalysis\n";
  $this->_printLogMessage($message);

  my $v0 = sqrt($beam->energy*$beam->energy - $beam->mass*$beam->mass)/$beam->energy;

  my %params = @_;
  my ($file, $delta) = ("./analysis" . "$ENV{HOST}", 0.0);

  if(defined $params{"print"})   { $file  = $params{"print"}; };

  open(ANALYSIS, ">$file") || die "==can't create $file";

  # if(defined $params{"dp/p"})   { $delta  = $params{"dp/p"}*$v0; }
  if(defined $params{"dp/p"})   { $delta  = $this->tpot2mad($this->{beam}, $params{"dp/p"});}
  my $orbit = new Pac::Position(); $orbit->de($delta);

  $message = "\n  closed orbit     " . time . "\n";
  $this->_printLogMessage($message);

	$this->{"orbit"} = $orbit;
  	$code->clorbit($orbit, $beam);

	my @ip = ("x[m]", "px", "y[m]", "py", "dE/p", "dp/p");
	printf ANALYSIS sprintf("Closed Orbit:\n\n");
	printf ANALYSIS sprintf("        %-11s %-11s %-11s %-11s %-9s %-9s\n", 
	$ip[0], $ip[1], $ip[2], $ip[3], $ip[4], $ip[5]);
	printf ANALYSIS sprintf("  %- 11.4e %- 11.4e %- 11.4e %- 11.4e %- 11.4e %- 11.4e\n",
	$orbit->x, $orbit->px, $orbit->y, $orbit->py, $orbit->de, $this->mad2tpot($this->{beam}, $orbit->de));

  $message = "\n  twiss parameters ". time . "\n";
  $this->_printLogMessage($message);

	my $rtwopi = 1./atan2(1,1)/8.;
  	my $chrom = new Pac::ChromData; 
  	$code->chrom($chrom, $beam, $orbit);

	my $twiss = $chrom->twiss();

	my @itw = ("beta", "alpha", "q", "d(dp/p)", "dd(dp/p)", "dq(dp/p)");
	printf ANALYSIS sprintf("\nTwiss Parameters:\n\n");
	printf ANALYSIS sprintf("      %-11s %-11s %-11s %-11s %-11s %-11s\n", 
	$itw[0], $itw[1], $itw[2], $itw[3], $itw[4], $itw[5]);
	printf ANALYSIS sprintf("x %- 11.4e %- 11.4e %- 11.4e %- 11.4e %- 11.4e %- 11.4e\n",
	$twiss->beta(0), $twiss->alpha(0), $twiss->mu(0)*$rtwopi, $v0*$twiss->d(0), $v0*$v0*$twiss->dp(0), 
	$v0*$chrom->dmu(0)*$rtwopi);
	printf ANALYSIS sprintf("y %- 11.4e %- 11.4e %- 11.4e %- 11.4e %- 11.4e %- 11.4e\n",
	$twiss->beta(1), $twiss->alpha(1), $twiss->mu(1)*$rtwopi, $v0*$twiss->d(1), $v0*$v0*$twiss->dp(1), 
	$v0*$chrom->dmu(1)*$rtwopi);

	my @itw = ("beta", "alpha", "q", "d(dE/p)", "dd(dE/p)", "dq(dE/p)");
	printf ANALYSIS sprintf("\nTwiss Parameters:\n\n");
	printf ANALYSIS sprintf("      %-11s %-11s %-11s %-11s %-11s %-11s\n", 
	$itw[0], $itw[1], $itw[2], $itw[3], $itw[4], $itw[5]);
	printf ANALYSIS sprintf("x %- 11.4e %- 11.4e %- 11.4e %- 11.4e %- 11.4e %- 11.4e\n",
	$twiss->beta(0), $twiss->alpha(0), $twiss->mu(0)*$rtwopi, $twiss->d(0), $twiss->dp(0), 
	$chrom->dmu(0)*$rtwopi);
	printf ANALYSIS sprintf("y %- 11.4e %- 11.4e %- 11.4e %- 11.4e %- 11.4e %- 11.4e\n",
	$twiss->beta(1), $twiss->alpha(1), $twiss->mu(1)*$rtwopi, $twiss->d(1), $twiss->dp(1), 
	$chrom->dmu(1)*$rtwopi);	

   $message = "\n  end            . " . time . "\n";
   $this->_printLogMessage($message);

  close(ANALYSIS);

}



sub map
{

  my $this   = shift;
  my %params = @_;
  my $order = $params{order};
  my $de = 0.0;  if(defined $params{"de"})   {$de = $params{"de"};} 

 my $message;
 $message = "\nmake map \n";   
 $this->_printLogMessage($message);

  my $map = new ALE::UI::SimpleMapping();
  $map->space($this, $order);

  $message = "start = " . time . "\n";
  $this->_printLogMessage($message);

  $map->map($this, $order, $de);

  $message = "end   = " . time . "\n";
  $this->_printLogMessage($message);

  if(defined $params{"print"}) { $this->{"map"}->write($params{"print"}); }

}

sub matrix
{   
  my $this   = shift;
  my %params = @_;

  my $order = 1;

  my $message;
  $message = "\nmake matrix \n";   
  $this->_printLogMessage($message);


  my $map = new ALE::UI::SimpleMatrix();
  $map->space($this, $order);

  my $delta = new Pac::Position;
  $delta->set(1.e-6, 1.e-6, 1.e-6, 1.e-6, 0.0, 1.e-6);


  $message = "start = " . time . "\n";
  $this->_printLogMessage($message);

  $map->map($this, $order, $delta);
 
  $message = "end   = " . time . "\n";
  $this->_printLogMessage($message);

  if(defined $params{"print"}) { $this->{"map"}->write($params{"print"}); }

}

# ************************************************************************
# Tracking 
# ************************************************************************

sub firstturn
{  
  my $this   = shift;
  my %params = @_;
  
  my ($particle, $file, $regex) = (0, "./firstturn", " ");

  my $message;
  $message = "\nfirstturn \n";   
  $this->_printLogMessage($message);
   
  if(defined $params{"particle"})   { $particle  = $params{"particle"}; }
  if(defined $params{"print"})      { $file      = $params{"print"}; }
  if(defined $params{"observe"})    { $regex     = $params{"observe"}; }

  # Prepare the bunch with one particle

  $this->{bunch} = new Pac::Bunch(1);
  my $p = new Pac::Position();
  $p->set(@{$particle});
  $this->{bunch}->position(0, $p);
 
  $message = "start = " . time . "\n";
  $this->_printLogMessage($message);

  my $tracking = new ALE::UI::SimpleTracking();
  $tracking->firstturn($file, $regex, $this);
 
  $message = "end   = " . time . "\n";
  $this->_printLogMessage($message);  
}

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

  my $service = new ALE::UI::SimpleTracking();
  $service->run($file, $turns, $step, $this, $bunch->{"bunch"}, $orbit);

  $message = "end   = " . time . "\n";
  $shell_there->_printLogMessage($message);  
 
}

# ****************************************
# Fitting & Correction
# ****************************************

sub hsteer
{  
   my ($this) = shift;

  my $message;
  $message = "\nhsteer \n";   
  $this->_printLogMessage($message);
  
  $message = "start = " . time . "\n";
  $this->_printLogMessage($message);

   $this->{code}->hsteer(@_, "beam" => $this->{"beam"});

  $message = "end   = " . time . "\n";
  $this->_printLogMessage($message);  
}

sub vsteer
{    
  my ($this) = shift;

  my $message;
  $message = "\nvsteer \n";   
  $this->_printLogMessage($message);
   
  $message = "start = " . time . "\n";
  $this->_printLogMessage($message);

   $this->{code}->vsteer(@_, "beam" => $this->{"beam"});

  $message = "end   = " . time . "\n";
  $this->_printLogMessage($message);  
}

sub tunethin
{  
   my ($this) = shift;

   my $message;
   $message = "\ntunethin \n";   
   $this->_printLogMessage($message);

  $message = "start = " . time . "\n";
  $this->_printLogMessage($message);

   $this->{code}->tunethin(@_, "beam" => $this->{"beam"});

  $message = "end   = " . time . "\n";
  $this->_printLogMessage($message);  
}


sub chromfit
{  

   my ($this) = shift;

   my $message;
   $message = "\nchromfit \n";   
   $this->_printLogMessage($message);

  $message = "start = " . time . "\n";
  $this->_printLogMessage($message);

   $this->{code}->chromfit(@_, "beam" => $this->{"beam"});

  $message = "end   = " . time . "\n";
  $this->_printLogMessage($message);  

}

sub decouple
{  
   my ($this) = shift;

   my $message;
   $message = "\ndecouple \n";   
   $this->_printLogMessage($message);


  $message = "start = " . time . "\n";
  $this->_printLogMessage($message);

   $this->{code}->decouple(@_, "beam" => $this->{"beam"});

  $message = "end   = " . time . "\n";
  $this->_printLogMessage($message);  

}




# ************************************************************************
# Survey
# ************************************************************************

sub survey
{ 

  my $this   = shift;
  my %params = @_;

   my $message;
   $message = "\nsurvey \n";   
   $this->_printLogMessage($message);

  my ($file, $regex)  = ("./survey", " ");
  if(defined $params{"print"})   { $file  = $params{"print"}; }
  if(defined $params{"elements"}) { $regex = $params{"elements"}; }

  $message = "start = " . time . "\n";
  $this->_printLogMessage($message);

  my $survey = new ALE::UI::SimpleSurvey($this->{shell});
  $survey->print($file, $regex, $this);

  $message = "end   = " . time . "\n";
  $this->_printLogMessage($message);  
}

# ************************************************************************
# Twiss
# ************************************************************************

sub twiss
{  
  
  my $this   = shift;
  my %params = @_;

   my $message;
   $message = "\ntwiss \n";   
   $this->_printLogMessage($message);

  my ($file, $regex, $couple)  = ("./twiss", " ", 0);
  if(defined $params{"print"})   { $file   = $params{"print"}; }
  if(defined $params{"elements"}) { $regex  = $params{"elements"}; }
  if(defined $params{"couple"})  { $couple = $params{"couple"}; }

  $message = "start = " . time . "\n";
  $this->_printLogMessage($message);

  my $twiss = new ALE::UI::SimpleTwiss($this->{shell});
  $twiss->print($file, $regex, $couple, $this);

  $message = "end   = " . time . "\n";
  $this->_printLogMessage($message);  

}


# ****************************************
# Direct access to UAL classes
# ****************************************

# Return the pointer to the SMF instance

sub smf
{
  my $this = shift;
  return $this->{shell}->smf;
}

# Return the pointer to the beam

sub beam
{
  my $this = shift;
  return $this->{"beam"};
}

# Return the pointer to the bunch

sub bunch
{
  my $this = shift;
  return $this->{"bunch"};
}

sub closedorbit
{
  my $this = shift;
  return $this->{"orbit"};
  # my $o = $this->{"orbit"};
  # my @clorbit = ($o->x, $o->px, $o->y, $o->py, $o->ct, $o->de);
  # return @clorbit;
}

# Return the pointer to the TEAPOT
sub teapot
{
  my $this = shift;
  return $this->{"code"};
}

# ****************************************
# Secondary methods
# ****************************************

sub _printLogMessage
{
  my ($this, $message) = @_;

  my $file = $this->{"log"};
  open(LOG, ">>$file") || die "can't create $file";
  printf LOG $message;
  close(LOG);    

}

sub tpot2mad
{
  my ($this, $beam, $dp) = @_;

  my ($e0, $p0, $m0, $p, $e);

  $e0 = $beam->energy;
  $m0 = $beam->mass;

  $p0 = $e0*$e0 - $m0*$m0;
  $p0 = sqrt($p0);

  $p = $p0*(1.0 + $dp);
  $e = sqrt($p*$p + $m0*$m0);

  return ($e - $e0)/$p0;
}

sub mad2tpot {
    my ($this, $beam, $de) = @_;
    my ($e0, $p0, $m0, $p, $e);

    $e0 = $beam->energy;
    $m0 = $beam->mass;

    $p0 = $e0*$e0 - $m0*$m0;
    $p0 = sqrt($p0);

    $e = $p0*$de + $e0;
    $p = sqrt(($e - $m0)*($e + $m0));
    return ($p - $p0)/$p0;  
}
1;

__END__


=head1

=begin html
<h1> Class <a href="./package.html"> ALE::UI</a>::Shell</h1>
<hr>
<h3> Extends: </h3>
The Shell class provides a  simple user-friendly (MAD+TEAPOT-specific) interface 
to UAL environment.
<hr>
<pre><h3>Sample Script:  <a href="./Shell.txt"> Shell.pl </a> </h3></pre>
<h3> Public Methods </h3>
<h4><i> Constructor </i></h4>
<ul>
<li> <b> new() </b>
<dl>
    <dt> Constructor.
</dl>
</ul>
<h4> <i>Accelerator description, selection, etc. </i> </h4>
<ul>
<li> <b> split(%parameters) </b>
<dl> 
    <dt>Splits selected elements into several thin multipoles.
    <dd><i>parameters{$pattern}</i> - an associative array of pattern for 
    selecting elements and TEAPOT split number (1 - IR, 2 - IR2, etc.). 
    <dt> Example :
    <dd> split("^(i\_qx|qx).*" => 1)
</dl>
<li> <b> use($lattice) </b>
<dl> 
    <dt>Selects an accelerator lattice for operations (Builds a lattice
    from a MAD line or MAD sequence with the same name).
    <dd><i>lattice</i> - a lattice name.
    <dt> Example :
    <dd> use("lhc")
</dl>
</ul>
<h4> <i>Analysis</i></h4>
<ul>
<li> <b> analysis(%parameters) </b>
<dl>
    <dt> Finds the closed orbit and performs twiss analysis of the machine.
    <dd><i>$parameters{print}</i> - an output file name (default: "./analysis"). 
    <dd><i>$parameters{delta}</i> - a momentum offset. 
    <dt> Example :
    <dd> analysis("print" => "./data/analysis.out", delta => 1.0e-3)
</dl>
</ul>
<h4> <i>Fitting</i></h4>
<ul>
<li> <b> hsteer(%parameters) </b>
<dl>
    <dt> Flattens the orbit horizontally (Delegates the request to the 
	corresponding TEAPOT command).
    <dd><i>$parameters{adjusters}</i> - a regular expression for selecting adjusters. 
    <dd><i>$parameters{detectors}</i> - a regular expression for selecting detectors. 
    <dt> Example :
    <dd> hsteer(adjusters => "^kickh\$", detectors => "^bpmh\$");
</dl>
<li> <b> vsteer(%parameters) </b>
<dl>
    <dt> Flattens the orbit verically (Delegates the request to the 
	corresponding TEAPOT command).
    <dd><i>$parameters{adjusters}</i> - a regular expression for selecting adjusters. 
    <dd><i>$parameters{detectors}</i> - a regular expression for selecting detectors. 
    <dt> Example :
    <dd> vsteer(adjusters => "^kickv\$", detectors => "^bpmv\$");
</dl>
<li> <b> tunethin(%parameters) </b>
<dl>
    <dt> Fits the tunes of the machine (Delegates the request to the 
	corresponding TEAPOT command).
    <dd><i>$parameters{bf}</i> - a regular expression for selecting focusing elements. 
    <dd><i>$parameters{bd}</i> - a regular expression for selecting defocusing elements. 
    <dd><i>$parameters{mux}</i> - a requested horizontal tune value. 
    <dd><i>$parameters{muy}</i> - a requested vertical tune value. 
    <dd><i>$parameters{method}</i> - a method to alter correction elements,
	 multiplicative ('*') or additive ('+') (default: '*').
    <dd><i>$parameters{numtries}</i> - the maximum number of iterations 
	for the fitting (default: 100).
    <dd><i>$parameters{tolerance}</i> - the maximum absolute value of 
	the difference between requested values and the fitted values 
	at convergence (default: 1.0e-6).
    <dt> Example :
    <dd> tunethin(bf => "^qf\$", bd => "^qd\$", mux => 28.19, muy => 29.18);
</dl>
<li> <b> chromfit(%parameters) </b>
<dl>
    <dt> Fits the chromaticity of the machine (Delegates the request 
	to the corresponding TEAPOT command).
    <dd><i>$parameters{bf}</i> - a regular expression for selecting focusing elements. 
    <dd><i>$parameters{bd}</i> - a regular expression for selecting defocusing elements. 
    <dd><i>$parameters{chromx}</i> - a requested horizontal  chromaticity value. 
    <dd><i>$parameters{chromy}</i> - a requested vertical chromaticity value. 
    <dd><i>$parameters{method}</i> - a method to alter correction elements,
	 multiplicative ('*') or additive ('+') (default: '*').
    <dd><i>$parameters{numtries}</i> - the maximum number of iterations 
	for the fitting (default: 10).
    <dd><i>$parameters{tolerance}</i> - the maximum absolute value of 
	the difference between requested values and the fitted values 
	at convergence (default: 1.0e-4).
    <dt> Example :
    <dd> chromfit(bf => "^sf\$", bd => "^sd\$", chromx => -3.0, chromy => -3.0)
</dl>
<li> <b> decouple(%parameters) </b>
<dl>
    <dt> Zeros two elements, E12 and E22, of the matrix E = B + bar(C), 
	as well as adjusting the tunes  (Delegates the request to the 
	corresponding TEAPOT command).
    <dd><i>$parameters{a11}</i> - a regular expression for selecting the 1st sextupole family. 
    <dd><i>$parameters{a12}</i> - a regular expression for selecting the 2nd sextupole family.
    <dd><i>$parameters{a13}</i> - a regular expression for selecting the 3rd sextupole family. 
    <dd><i>$parameters{a14}</i> - a regular expression for selecting the 4th sextupole family. 
    <dd><i>$parameters{bf}</i> - a regular expression for selecting focusing quadrupoles. 
    <dd><i>$parameters{bd}</i> - a regular expression for selecting defocusing quadrupoles.
    <dd><i>$parameters{mux}</i> - a requested horizontal tune value. 
    <dd><i>$parameters{muy}</i> - a requested vertical tune value. 
    <dt> Example :
    <dd> decouple(a11 => "^sqsk6\$", a12 => "^sqsk8\$", a13 => "^sqsk12\$", a14 => "^sqsk2\$", 
	bf => "^qf\$", bd => "^qd\$", mux =>  28.19, muy => 29.18)
</dl>
</ul>
<h4> <i>Survey (Accelerator Geometry) </i></h4>
<ul>
<li> <b> survey(%parameters) </b>
<dl>
    <dt> Calculates an accelerator geometry (survey) and prints results 
    for selected elements.
    <dd><i>$parameters{print}</i> - an output file name (default: "./survey"). 
    <dd><i>$parameters{observe}</i> - a regular expression for selecting 
	elements (default: " ").
    <dt> Example :
    <dd> survey(print => "./data/survey.out", observe => "ip")
</dl>
</ul>
<h4> <i>Tracking </i></h4>
<ul>
<li> <b> beam(%parameters) </b>
<dl>
    <dt> Defines beam parameters (Initializes the Pac::BeamAttributes object).
    <dd><i>$parameters{energy}</i> - beam energy (GeV, default: infinity).
    <dd><i>$parameters{mass}</i>   - particle mass (GeV, default: 0.93828).
    <dd><i>$parameters{charge}</i> - particle charge (default: 1).
    <dt> Example:
    <dd> beam(energy => 42.e+2, mass => 0.93828, charge => 1)
</dl>
<li> <b> start(@positions) </b>
<dl>
    <dt> Defines particles' initial coordinates, displacements from 
	the reference orbit (Initializes the Pac::Bunch object)
    <dd><i>positions</i> - an array of MAD particle coordinates, 
	[x, px/p0, y, py/p0, dt, de/p0].
    <dt> Example:
    <dd> start([1.e-5, 0.0, 1.e-5, 0.0, 0.0, 1.e-5],
	[2.e-5, 0.0, 2.e-5, 0.0, 0.0, 1.e-5], )
</dl>
<li> <b> firstturn(%parameters) </b>
<dl>
    <dt> Tracks a particle for an one turn and prints results at 
	selected elements.
    <dd><i>$parameters{print}</i> - an output file name 
	(default: "./firstturn"). 
    <dd><i>$parameters{observe}</i> - a regular expression 
	for selecting elements (default: " ").
    <dt> Example :
    <dd> firstturn(print => "./data/firstturn.out", observe => "ip")
</dl>
<li> <b> run(%parameters) </b>
<dl>
    <dt> Tracks particles and provides the file8 output for offline post-processing
	tools such as tealeaf, turnplot, etc..
    <dd><i>$parameters{print}</i> - an output file name (default: "./fort.8"). 
    <dd><i>$parameters{orbit}</i> - orbit coordinates (default: [0,0,0,0,0,0]).
    <dd><i>$parameters{turns}</i> - number of turns (default: 1). 
    <dd><i>$parameters{step}</i>  - printing after "step" turns (default: 1).
    <dt> Example :
    <dd> run("print" => "./out/8/file", turns => 1);
</dl>
</ul>
<h4> <i>Mapping </i></h4>
<ul>
<li> <b> space($shell, $order) </b>
<dl>
    <dt> Defines global parameters of 6D maps. 
    <dd><i>order</i>   - the maximum order of Taylor maps.  
</dl>
<li> <b>map(%parameters) </b>
<dl>
    <dt> Makes a one-turn 6D map.
    <dd><i>$parameters{order}</i>  - the map order.
    <dd><i>$parameters{print}</i>  - an output file name.
    <dt> Example:
    <dd> map(order => 2, print => "./data/map.out")
</dl>
<li> <b>matrix(%parameters) </b>
<dl>
    <dt> Makes a one-turn 6D linear matrix (FTPOT approach).
    <dd><i>$parameters{print}</i>  - an output file name.
    <dt> Example:
    <dd> matrix(print => "./data/matrix.out")
</dl>
</ul>
<h4> <i>I/O methods </i></h4>
<ul>
<li> <b> read(%parameters) </b>
<dl>
    <dt>Reads accelerator data from local sources (Delegates
    <i> parameters </i> directly to the PAC::FTPOT::Shell::restore method).
    <dd><i>$parameters{files}</i> - a pointer to array of MAD file names.
    <dt> Example :
    <dd> read(files => ["./data/lhc.Kinj", "./data/lhc.seq"])
</dl>
<li> <b> write(%parameters) </b>
<dl>
    <dt>Writes SMF data to a MAD file (Delegates <i> parameters </i> 
    directly to the PAC::FTPOT::Shell::store method).
    <dd><i>$parameters{file}</i> - a MAD file name.
    <dt> Example :
    <dd> write(file => "./data/shell.out")
</dl>
</ul> 
<hr>

=end html
