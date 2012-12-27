#!/usr/local/bin/perl

# Resort all lines in "./NikolayOut" into groups with all turn of particle 0
# then two blank lines (the required gnuplot data separator) then all turns for
# particle 1 then two blanks, and so on. 

# For bunches with other than 21 particles change "$NUMPARTICLES".

# perl ~/perl/SplitNikolayOut.pl NikolayOut >! InputForGnuplot

$, = ' ';		# set output field separator
$\ = "\n";		# set output record separator

$NUMPARTICLES = 21;
while (<>) {
    chop;	# strip record separator
    $LineTemp = $_;
    @Fld = split(' ', $LineTemp, 9999);
    $ParamsByPartNum[$Fld[0]] .= $LineTemp . "\n";
}

for ($i=0; $i<$NUMPARTICLES; $i++) {
    print $ParamsByPartNum[$i] . "\n";
}
