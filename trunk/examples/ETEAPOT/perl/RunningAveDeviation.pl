#!/usr/local/bin/perl

# perl RunningAveDeviation.pl NikolayOut >! NikRunningAveDev
# perl ~/perl/SplitNikolayOut.pl NikRunningAveDev >! IG

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
    $AverageWeight=0.2;
    if($TurnNum eq 0){
	for($j=2; $j<$NUMCOLS; $j++) {
	    $InitField[$j][$PartNum] = $Fld[$j];
	}
	for($j=2; $j<$NUMCOLS; $j++) {
	    $OldField[$j][$PartNum] = $Fld[$j];
	}
    }
    if($TurnNum > 0){
	if($#Fld > 2) {
	    for($j=2; $j<$NUMCOLS; $j++) {
		$RunAveDev[$j][$PartNum] = ($AverageWeight*$OldFld[$j]+$Fld[$j])/(1+$AverageWeight) - $InitField[$j][$PartNum];
		$OldFld[$j] = $Fld[$j];
	    }
	    print $PartNum." ".$TurnNum." ".$RunAveDev[2][$PartNum]." ".$RunAveDev[3][$PartNum]." ".$RunAveDev[4][$PartNum]." ".$RunAveDev[5][$PartNum]." ".$RunAveDev[6][$PartNum]." ".$RunAveDev[7][$PartNum]." ".$RunAveDev[8][$PartNum]." ".$RunAveDev[9][$PartNum]." ".$RunAveDev[10][$PartNum]." ".$RunAveDev[11][$PartNum];}
    }
}

