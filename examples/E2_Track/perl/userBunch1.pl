#!/usr/bin/perl

# Resort all lines in "./JDTOut" into groups with all turn of particle 0
# then two blank lines (the required gnuplot data separator) then all turns for
# particle 1 then two blanks, and so on.

# For bunches with other than 2 particles change "$NUMPARTICLES".

$, = ' ';        # set output field separator
$\ = "\n";        # set output record separator

$DummyLine = "0 0 0 0 0 0 0 0 0 0 0 0 # Dummy Line to adjust indexing\n\n";
print $DummyLine;

$NUMPARTICLES = 2;
while (<>) {
    chop;    # strip record separator
    $LineTemp = $_;
    @Fld = split(' ', $LineTemp, 9999);
    $ParamsByPartNum[$Fld[0]] .= $LineTemp . "\n";
}

for ($i=1; $i<=$NUMPARTICLES; $i++) {
    print $ParamsByPartNum[$i] . "\n";
}
