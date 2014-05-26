#!/usr/local/bin/perl

# perl SumOverBunch.pl NikolayOut >! CentroidsTracked

$, = ' ';		# set output field separator
$\ = "\n";		# set output record separator

$NUMPARTS = 21;
$NUMCOLS = 11;
$TurnNum=0;
while (<>) {
    chop;	# strip record separator
    $LineTemp = $_;
    @Fld = split(' ', $LineTemp, 9999);
    $ParticleNum++;
    if($ParticleNum < $NUMPARTS) {
	for($j=2; $j<$NUMCOLS; $j++) {
	    $Accum[$j] += $Fld[$j]/$NUMPARTS;
	}
    }
    else{
	print "0 ".$TurnNum." ".$Accum[2]." ".$Accum[3]." ".$Accum[4]." ".$Accum[5]." ".$Accum[6]." ".$Accum[7]." ".$Accum[8]." ".$Accum[9]." ".$Accum[10]." ".$Accum[11];
	$TurnNum++;
	for($j=2; $j<$NUMCOLS; $j++) {
	    $Accum[$j] = 0;
	}
	$ParticleNum = 0;
    }
}
