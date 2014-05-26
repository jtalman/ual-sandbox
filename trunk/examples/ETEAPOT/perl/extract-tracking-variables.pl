#!/usr/local/bin/perl

# perl ~/perl/extract-tracking-variables.pl OUT >! ETEAPOT_MltTurn-track.data

$, = ' ';		# set output field separator
$\ = "\n";		# set output record separator

$PARTICLE = ONE;
$ielem = 0;
$begun = 0;
while (<>) {
    chop;	# strip record separator
    $LineTemp = $_;
    if( ($LineTemp =~ /^JDT\-pOf/) && ($LineTemp =~ /$PARTICLE/) ){
	@Fld = split(' ', $LineTemp, 9999);
        if ( $begun == 0 ) {
	    $ielem++;
	    $begun = 1;
	}
	if   ($Fld[0]  =~ /0\+1$/ ) {$p0[$ielem] = $Fld[1];}
	elsif ($Fld[0] =~ /1\+1$/ ) {$p1[$ielem] = $Fld[1];}
	elsif ($Fld[0] =~ /2\+1$/ ) {$p2[$ielem] = $Fld[1];}
	elsif ($Fld[0] =~ /3\+1$/ ) {$p3[$ielem] = $Fld[1];}
	elsif ($Fld[0] =~ /4\+1$/ ) {$p4[$ielem] = $Fld[1];}
	elsif ($Fld[0] =~ /5\+1$/ ) {$p5[$ielem] = $Fld[1];}
	else { $begun = 0; }
    }
    else {$begun = 0;}
}
$nelem = $ielem;

for ( $ielem=1; $ielem<$nelem; $ielem++ ){
    $LineOut = $ielem." ".$p0[$ielem]." ".$p1[$ielem]." ".$p2[$ielem]." ".$p3[$ielem]." ".$p4[$ielem]." ".$p5[$ielem];
    print $LineOut;
}
