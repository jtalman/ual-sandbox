#!/usr/local/bin/perl

# perl ~/perl/compare-oldANDnewTOF.pl oldTOF-OUT >! oldTOF.data
# perl ~/perl/compare-oldANDnewTOF.pl newTOF-OUT >! newTOF.data

$, = ' ';		# set output field separator
$\ = "\n";		# set output record separator

$ielem = 0;
$begun = 0;
while (<>) {
    chop;	# strip record separator
    $LineTemp = $_;
    if( $LineTemp =~ /^JDTRT-pOf4/ ){
	@Fld = split(' ', $LineTemp, 9999);
        if ( $begun == 0 ) {
	    $ielem++;
	    $begun = 1;
	}
	if    ($Fld[0] =~ /^JDTRT-pOf4DEV/ ) {$p0[$ielem] = $Fld[1];}
	elsif ($Fld[0] =~ /^JDTRT-pOf4CUM/ ) {$p1[$ielem] = $Fld[1];}
	elsif ($Fld[0] =~ /^JDTRT-pOf4theta0/ ) {$p2[$ielem] = $Fld[1];}
	else { $begun = 0; }
    }
    else { $begun = 0; }
}
$nelem = $ielem;

for ( $ielem=1; $ielem<$nelem; $ielem++ ){
    $LineOut = $ielem." ".$p0[$ielem]." ".$p1[$ielem]." ".$p2[$ielem];
    print $LineOut;
}
