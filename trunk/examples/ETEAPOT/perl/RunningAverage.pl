#!/usr/local/bin/perl

# perl RunningAverage.pl NikolayOut >! NikolayRunningAve
# perl ~/perl/SplitNikolayOut.pl NikolayRunningAve >! IG

$, = ' ';		# set output field separator
$\ = "\n";		# set output record separator

$NUMPARTS = 21;
$NUMCOLS = 11;
while (<>) {
    chop;	# strip record separator
    $LineTemp = $_;
    @Fld = split(' ', $LineTemp, 9999);
    $PartNum=$Fld[0];
    $TurnNum=$Fld[1];
    if($#Fld > 2) {
	for($j=2; $j<$NUMCOLS; $j++) {
	    $RunAv[$j][$PartNum] += $Fld[$j]/($TurnNum+1);
	}
	print $PartNum." ".$TurnNum." ".$RunAv[2][$PartNum]." ".$RunAv[3][$PartNum]." ".$RunAv[4][$PartNum]." ".$RunAv[5][$PartNum]." ".$RunAv[6][$PartNum]." ".$RunAv[7][$PartNum]." ".$RunAv[8][$PartNum]." ".$RunAv[9][$PartNum]." ".$RunAv[10][$PartNum]." ".$RunAv[11][$PartNum];
    }
}
