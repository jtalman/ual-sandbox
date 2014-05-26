#!/usr/local/bin/perl

# perl ~/perl/ctTOdct-Xi-ETEAPOT.pl Compare-Xi-ETEAPOT-bends.data >! Compare-Xi-ETEAPOT-diffs.data

$, = ' ';		# set output field separator
$\ = "\n";		# set output record separator

$ctXiPrev=0.0;
$ctETEAPOTPrev=0.0;

while (<>) {
    chop;	# strip record separator
    $LineTemp = $_;
    @Fld = split(' ', $LineTemp, 9999);
    $dctXi = $Fld[4]-$ctXiPrev;
    $ctXiPrev = $Fld[4];
    $dctETEAPOT = $Fld[7]-$ctETEAPOTPrev;
    $ctETEAPOTPrev = $Fld[7];
    $LineOut = $LineTemp." ".$dctXi." ".$dctETEAPOT;
    print $LineOut;
}
