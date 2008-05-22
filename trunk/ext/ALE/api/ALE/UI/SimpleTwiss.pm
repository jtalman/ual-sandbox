package ALE::UI::SimpleTwiss;

use strict;
use Carp;

use File::Basename;

my $L_ = 0;
my $DIM = 6;

sub new
{
    my ($type, $shell) = @_;
    my $this = {};

    $shell->map->attribKeyFromString("l", \$L_);

    return bless $this, $type;
}

sub print
{
    my ($this, $file, $regex, $couple, $shell) = @_;

    my $lattice = $shell->{lattice};
    my $code = $shell->{code};
    my $beam = $shell->{beam};

    my $dir = dirname($file);

    # Closed orbit

    my $orbit = new Pac::Position();
    $code->clorbit($orbit, $beam);

    # One-Turn Map 

    my $oneTurnMap = new Pac::TMap($DIM);

    $oneTurnMap->refOrbit($orbit);	
    $code->map($oneTurnMap, $beam, 1); 

    my $eigenMap = new Pac::TMap($DIM);
    
    open(TWISS, ">$file") || die "can't create file(twiss)";

    my @columns = ("#", "name", "suml", "betax", "alfax", "qx", "dx", "betay", "alfay", "qy", "dy");    

    print TWISS "------------------------------------------------------------";
    print TWISS "------------------------------------------------------------\n"; 
    my $output = sprintf("%-5s %-10s   %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n", 
	$columns[0],  $columns[1], $columns[2], $columns[3],  $columns[4],
	$columns[5],  $columns[6], $columns[7], $columns[8], $columns[9], $columns[10]);
    print TWISS $output;

    print TWISS "------------------------------------------------------------";
    print TWISS "------------------------------------------------------------\n"; 

    my $twiss = new Pac::TwissData; 
    my $chrom = new Pac::ChromData;
    my ($i, $le, $suml, $bName) = (0, 0, 0, " ");

    my $rtwopi = 1./atan2(1,1)/8.;   

    if($couple) {
	$code->transformOneTurnMap($eigenMap, $oneTurnMap);
        $eigenMap->write($dir . "/eigenMap");
	$code->eigenTwiss($twiss, $eigenMap);
    }
    else{
	# $code->eigenTwiss($twiss, $oneTurnMap);
        $code->chrom($chrom, $beam, $orbit);
        $twiss = $chrom->twiss();
        $twiss->mu(0, 0.0);
        $twiss->mu(1, 0.0);

    }	

    $output = sprintf("%5d %-10s %15.7e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e\n", 
	$i, $bName, $suml, 
	$twiss->beta(0), $twiss->alpha(0), $twiss->mu(0)*$rtwopi, $twiss->d(0),
	$twiss->beta(1), $twiss->alpha(1), $twiss->mu(1)*$rtwopi, $twiss->d(1));
    print TWISS $output;

    # my $sectorMap = new Pac::TMap($DIM); 
    # $sectorMap->refOrbit($orbit);  

    my ($mux, $muy) = (0.0, 0.0);
 
    for($i=0; $i < $lattice->size; $i++){
	$le = $lattice->element($i);    
    
	if($le->genName =~ $regex) {
    	   $output = sprintf("%5d %-10s %15.7e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e\n", 
		$i, $le->genName(), $suml, 
		$twiss->beta(0), $twiss->alpha(0), $twiss->mu(0)*$rtwopi, $twiss->d(0),
		$twiss->beta(1), $twiss->alpha(1), $twiss->mu(1)*$rtwopi, $twiss->d(1));		
	   print TWISS $output;

	}

        my $sectorMap = new Pac::TMap($DIM); 
        $sectorMap->refOrbit($orbit);  

	$code->trackMap($sectorMap, $beam, $i, $i+1);
	$code->trackClorbit($orbit, $beam, $i, $i+1);

    	if($couple) {
		$code->transformSectorMap($eigenMap, $oneTurnMap, $sectorMap);
		$code->trackEigenTwiss($twiss, $eigenMap);
    	}
   	 else{
		$code->trackTwiss($twiss, $sectorMap);

		if(($twiss->mu(0) - $mux) < 0.0) { $twiss->mu(0, $twiss->mu(0) + 1.0); }
		$mux = $twiss->mu(0);

		if(($twiss->mu(1) - $muy) < 0.0) { $twiss->mu(1, $twiss->mu(1) + 1.0); }
		$muy = $twiss->mu(1);
    	}

	
	$suml += $le->get($L_); 
    }   
   
    print TWISS "------------------------------------------------------------";
    print TWISS "------------------------------------------------------------\n"; 
    my $output = sprintf("%-5s %-10s   %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n", 
	$columns[0],  $columns[1], $columns[2], $columns[3],  $columns[4],
	$columns[5],  $columns[6], $columns[7], $columns[8], $columns[9], $columns[10]);
    print TWISS $output;

    print TWISS "------------------------------------------------------------";
    print TWISS "------------------------------------------------------------\n";  
  
    # $oneTurnMap->write($dir . "/oneTurnMap");
    # $sectorMap->write($dir . "/sectorMap");
    close(TWISS);
}

1;

__END__


=head1

=begin html
<h1> Class <a href="./package.html"> ALE::UI</a>::SimpleSurvey </h1>
<hr>
<h3> Extends: </h3>
The SimpleSurvey class calculates an accelerator geometry (survey) and writes
the results to a file. 
<hr>
<h3> Public Methods </h3>
<ul>
<li> <b> new($shell) </b>
<dl>
    <dt> Constructor.
    <dd><i>shell</i> - a pointer to a PAC::MAD::Shell instance.
</dl>
<li> <b> print($file, $pattern, $shell) </b>
<dl>
    <dt> Calculates an accelerator survey for selected elements and 
	writes a simple output.
    <dd><i>file</i>    - a output file.
    <dd><i>pattern</i> - a regular expression for selecting elements. 
    <dd><i>shell</i>   - a pointer to a ALE::UI::Shell instance.  
</dl>
</ul> 
<hr>

=end html



