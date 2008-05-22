package ALE::UI::SimpleTracking;

use strict;
use Carp;

sub new
{
    my $type = shift;
    my $this = {};
    return bless $this, $type;
}


sub firstturn
{
  my ($this, $file, $regex, $shell) = @_;

  my $lattice = $shell->{lattice};
  my $code    = $shell->{code};
  my $beam    = $shell->{beam};
  my $bunch   = $shell->{bunch};

  open(FIRSTTURN, ">$file") || die "can't create $file";

  printf FIRSTTURN " # First turn information at every \n";
#  printf FIRSTTURN "     #i element              x        vxbyc          y        vybyc          ct \n";
   printf FIRSTTURN "     #i element              x          px           y          py           ct \n";


  $bunch = $this->_makebunch($beam, $bunch);
  # $this->_tpot2mad($bunch);

  my $pout   = $bunch->position(0);
  my $v0     = sqrt($bunch->energy*$bunch->energy - $bunch->mass*$bunch->mass)/$bunch->energy;

  my $de = 0.0;
  my $revfreq = $this->_revfreq($shell, $beam, $de);
  $bunch->revfreq($revfreq);

  my ($i, $size, $elname) = (0, $lattice->size, "");
  my $counter = 0;
  for($i=0; $i < $size; $i++){
    $code->track($bunch, $i, $i + 1);
    $elname = $lattice->element($i)->genName;
    if ($elname eq "") { $elname = $counter++;}
    if($elname =~ $regex) {
       $pout = $bunch->position(0);
       printf FIRSTTURN sprintf("%6d  %-15s %- 16.9e %- 16.9e %- 16.9e %- 16.9e %- 16.9e %- 16.9e\n", 
#	                      $i, $elname, $pout->x,  $pout->px*$v0, $pout->y,  $pout->py*$v0, $pout->ct);
	                      $i, $elname, $pout->x,  $pout->px, $pout->y,  $pout->py, $pout->ct, $pout->de);
    }
  }

  $pout   = $bunch->position(0);
  $elname = "End";
  printf FIRSTTURN sprintf("%6d  %-15s %- 16.9e %- 16.9e %- 16.9e %- 16.9e %- 16.9e\n", 
#	        $i, $elname, $pout->x,  $pout->px*$v0, $pout->y,  $pout->py*$v0, $pout->ct);
	        $i, $elname, $pout->x,  $pout->px, $pout->y,  $pout->py, $pout->ct);

  close(FIRSTTURN);
}


sub run
{
  my ($this, $file, $turns, $step, $shell, $bunch, $orbit) = @_;

  my $vertrk = "version     4.0   tracking  ";
  my $seed   = "     60933";
  my $title  = "!  TEAPOT Tracking Output (fort.8 file)";
# Return todays date in alternate form (ie: ##/##/##) for concatenating onto
# filenames.
  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime($^T);
  my $daynum = ("01","02","03","04","05","06","07","08","09","10","11","12",
		"13","14","15","16","17","18","19","20","21","22","23","24",
		"25","26","27","28","29","30","31")[$mday-1];
  my $monnam = ("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct",
		"Nov","Dec")[$mon];
  my $weknam = ("Mon","Tue","Wed","Thu","Fri","Sat","Sun")[$wday-1];
#    print $sec," ",$min," ",$hour," ",$daynum," ",$monnam," ",$year," ";
#    print $wday, "\n";

#    $numdate = "$year/$monnum/$daynum\n";
  my $namdate = "$weknam$monnam$daynum";
  my $timstmp = "$hour:$min:$sec";

  open(FORT8, ">$file") || die "can't create file($file)";

  my $code    = $shell->{code};
  my $beam    = $shell->{beam};
  my $np      = $bunch->size();

  # Print Twiss parameters

  my $rtwopi = 1./atan2(1,1)/8.;
  my $twiss  = new Pac::TwissData;

  $code->twiss($twiss, $beam, $orbit);

  print FORT8 $vertrk,$namdate,"  ",$timstmp,$seed,"\n";
  print FORT8 $title,"\n";
  print FORT8 "   ",$np,"      ",$turns,"  ";

  my $output = sprintf("%- 15.9E %- 15.9E",
                       $twiss->beta(0),  $twiss->beta(1));
  print FORT8 $output, "\n";

  $output = sprintf("%- 15.9E %- 15.9E %- 15.9E %- 15.9E",
                     $twiss->alpha(0), $twiss->alpha(1),
                     $twiss->mu(0)*$rtwopi, $twiss->mu(1)*$rtwopi);
  print FORT8 $output, "\n";

  my $particle = new Pac::Bunch(1);
#  $particle->energy($beam->energy);
#  $particle->charge($beam->charge);
#  $particle->mass($beam->mass);

  my $revfreq = $this->_revfreq($shell, $beam, $orbit->de); 

  my ($p, $t, $turn, $pout, $dpbyp);
  for($p = 0; $p < $np; $p++){

	$pout = $bunch->position($p);

	$turn = 0;
	$output = sprintf("%7d %- 15.9E %- 15.9E \n %- 15.9E %- 15.9E %- 15.9E %- 15.9E",
                $turn, $pout->x,  $pout->px, $pout->y, $pout->py, $pout->ct, $pout->de);
		print FORT8 $output, "\n";

        # $pout->set($pout->x + $orbit->x, $pout->px + $orbit->px, $pout->y + $orbit->y, $pout->py + $orbit->py,
	#	$pout->ct, $this->_dp2de($beam, $pout->de + $orbit->de));
	$pout->set($pout->x + $orbit->x, $pout->px + $orbit->px, $pout->y + $orbit->y, $pout->py + $orbit->py,
		$pout->ct, $pout->de + $orbit->de);
	$particle->position(0, $pout);

	# Teapot changes beam attributes during tracking because of rf effects

  	$particle->energy($beam->energy);
  	$particle->charge($beam->charge);
  	$particle->mass($beam->mass);
        $particle->revfreq($revfreq);

	for($t = 0; $t < $turns; $t += $step){

		$code->track($particle, $step);
                # $pout->set(-$orbit->x, -$orbit->px,-$orbit->y, -$orbit->py, 0.0,
                #           -$this->_dp2de($beam, $orbit->de));
                $pout->set(-$orbit->x, -$orbit->px,-$orbit->y, -$orbit->py, 0.0, -$orbit->de);
		$pout += $particle->position(0);

		$turn = $t + $step;
		if($particle->flag(0) <= 0) {
			$dpbyp = $this->_de2dp($beam, $pout->de);
			$output = sprintf("%7d %- 15.9E %- 15.9E %- 15.9E %- 15.9E %- 15.9E %- 15.9E",
		        	$turn, $pout->x,  $pout->px, $pout->y, $pout->py, $pout->ct, $dpbyp);
			print FORT8 $output, "\n";
		}
		else {
			print FORT8 "     -1  0.000000000E+00  0.000000000E+00  0.000000000E+00 " . 
			"0.000000000E+00  0.000000000E+00  0.000000000E+00\n";
			$t = $turns;
			$particle->flag(0, 0);
		}
	}

	$bunch->position($p, $particle->position(0));
	$bunch->flag($p, $particle->flag(0));

  }

  close(FORT8);

}


sub _makebunch
{
  my ($this, $beam, $b)  = @_;

  my $size  = $b->size;
  my $bunch = new Pac::Bunch($size);

  $bunch->energy($beam->energy);
  $bunch->charge($beam->charge);  
  $bunch->mass($beam->mass);

  my $i;
  for($i=0; $i < $size; $i++) {
    $bunch->position($i, $b->position($i));
  }

  return $bunch;
}

sub _de2dp {
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

sub _dp2de
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

sub _tpot2mad
{
  my ($this, $bunch) = @_;

  my ($position, $e0, $p0, $m0, $p, $e, $i);

  $e0 = $bunch->energy;
  $m0 = $bunch->mass;

  $p0 = $e0*$e0 - $m0*$m0;
  $p0 = sqrt($p0);
  for($i = 0; $i < $bunch->size; $i++){

        $position = $bunch->position($i);

	$p = $p0*(1.0 + $position->de);
	$e = sqrt($p*$p + $m0*$m0);
	$position->de(($e - $e0)/$p0); 

        $bunch->position($i, $position);
  }   
}

# sub _revfreq
# {
#  my ($this, $shell, $v0) = @_;
#  my $lattice = $shell->{lattice};
#  my $code    = $shell->{code};
#  my $beam    = $shell->{beam};
#
#  my $survey = new Pac::SurveyData;
#  $code->survey($survey, 0,  $lattice->size);
#  my $suml = $survey->suml;
#
#  my $revfreq = $v0*2.99792458e+8/$suml;
#  print "suml = ", $suml, " revfreq = ", $revfreq, "\n";
#  return $revfreq;
# }

sub _revfreq
{
  my ($this, $shell, $beam, $de) = @_;
  my $lattice = $shell->{lattice};
  my $code    = $shell->{code};
	
  my $message;
  $message ="\n_revfreq";
  $shell->_printLogMessage($message);


  my $v0 = sqrt($beam->energy*$beam->energy - $beam->mass*$beam->mass)/$beam->energy;

  # Suml
  my $survey = new Pac::SurveyData;
  $code->survey($survey, 0,  $lattice->size);
  my $suml   = $survey->suml;

  # Clorbit
  my $orbit = new Pac::Position(); $orbit->de($de);
  $code->clorbit($orbit, $beam);
  $code->trackClorbit($orbit, $beam, 0, $lattice->size);

  # excess circumference
  my $excess =  -$orbit->ct; 
  my $revfreq = $v0*2.99792458e+8/($suml + $excess);

  $message =    "\n suml                   = " . $suml . 
                "\n excess circumference   = " . $excess .
                "\n revfreq                = " . $revfreq ;
  $shell->_printLogMessage($message);

  return $revfreq;
}

1;

__END__

=head1

=begin html
<h1> Class <a href="./package.html"> ALE::UI</a>::SimpleTracking</h1>
<hr>
<h3> Extends: </h3>
The SimpleTracking class performs element-by-element tracking and 
writes results to a file. 
<hr>
<h3> Public Methods </h3>
<ul>
<li> <b> new() </b>
<dl>
    <dt> Constructor.
</dl>
<li> <b> firstturn($file, $pattern, $shell) </b>
<dl>
    <dt> Prints a first turn track output information for selected elements. 
    <dd><i>file</i>    - a output file.
    <dd><i>pattern</i> - a regular expression for selecting elements. 
    <dd><i>shell</i>   - a pointer to a ALE::UI::Shell instance.  
</dl>
</ul> 
<hr>

=end html
