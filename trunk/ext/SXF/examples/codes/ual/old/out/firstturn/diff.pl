use Carp;

if($#ARGV < 0) { croak "Usage: perl diff.pl <mad> <teapot> \n"; }

# Define tolerance

my @tolerence = (1.0e-15, 1.0e-15, 1.0e-15, 1.0e-15);

# Read files

my $header;

open(mad, $ARGV[0]) || die "can't create mad file: $ARGV[0]";
my @lines1 = <mad>;
close(mad);

$header = shift @lines1;

open(teapot, $ARGV[1] ) || die "can't create teapot file: $ARGV[1]";
my @lines2 = <teapot>;
close(teapot); 

$header = shift @lines2;
$header = shift @lines2;

# Compare lines.

open(tracking, ">tracking.diff") || die "can't create tracking.diff";

my ($counter1, $name1, $line1, $space2, $counter2, $name2, $line2, $flag); 
my @p1 = (0.0, 0.0, 0.0, 0.0);
my @p2 = (0.0, 0.0, 0.0, 0.0);
my @dp = (0.0, 0.0, 0.0, 0.0);

my $counter = 0;

while(@lines1){
 
    $counter++;
    $line1 = shift @lines1;
    $line1 =~ s/D/e/g;
    ($name1, $p1[0], $p1[1], $p1[2], $p1[3]) = split  /\s+/, $line1;
    $line2 = shift @lines2;
    ($space2, $counter2, $name2, $p2[0], $p2[1], $p2[2], $p2[3]) = split /\s+/, $line2;

    $flag = 0;
    for($i = 0; $i < 4; $i++) {
	$dp[$i] =  $p1[$i] - $p2[$i];
	if(abs($dp[$i]) >  $tolerence[$i]) { $flag++;} 
    }    

    if($flag > 0) {
	$output = sprintf("%7d %10s %- 15.9E %- 15.9E %- 15.9E %- 15.9E", 
			  $counter, $name1, $dp[0], $dp[1], $dp[2], $dp[3]);
	print tracking $output, "\n";	
    }
}

close(tracking);


