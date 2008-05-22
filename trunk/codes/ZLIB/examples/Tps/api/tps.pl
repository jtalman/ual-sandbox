# -------------------------------------------------------------------
# Set up the UAL/ZLIB environment. The environment variable UAL_ZLIB
# is defined in the setup-ual file.
# -------------------------------------------------------------------

use lib ("$ENV{UAL_ZLIB}/api");
use Zlib::Tps;

# -------------------------------------------------------------------
# Create and initialize the space of the Tps objects.
# -------------------------------------------------------------------

my $dimension = 6;
my $maxOrder  = 6;

my $space = new Zlib::Space($dimension, $maxOrder);

# -------------------------------------------------------------------
# Create the Tps instance.
# -------------------------------------------------------------------

my $tps = new Zlib::Tps();

# -------------------------------------------------------------------
# Set Tps data members (order & coefficients).
# -------------------------------------------------------------------

$tps->order($maxOrder);

for($i = 0; $i < $tps->size; $i++) { $tps->value($i, ($i + 1.)*0.01); }

# -------------------------------------------------------------------
# Copy "tps" data into the "copy" object. In Perl, the copy operator  
# can not be overloaded and has to be replaced by other operators.
# $copy refers to the new Tps instance created by the addition operator.
# -------------------------------------------------------------------

my $copy = $tps + 0.0;

# -------------------------------------------------------------------
# Call the addition operators with Tps objects and numbers. 
# $sum refers to the new Tps instance created by the addition operator.
# -------------------------------------------------------------------

my $sum = $copy + $tps + 1.0;

# -------------------------------------------------------------------
# Call the subtraction operators with Tps objects and numbers.
# $diff refers to the new Tps instance created by the subtraction 
# operator.
# -------------------------------------------------------------------

my $diff = $copy - $tps - 2.0;

# -------------------------------------------------------------------
# Call the multiplication operator with Tps objects and numbers.
# $mult refers to the new Tps instance created by the multiplication 
# operator.
# -------------------------------------------------------------------

print "Multiplication \n";

my $mult;

print "start = ", time, "\n";
for($i = 0; $i < 2000; $i++) { $mult = $sum*$sum;  }
print "finish = ", time, "\n";

$mult = $mult*1.5;

# -------------------------------------------------------------------
# Call the division operator with Tps objects and numbers.
# $div refers to the new Tps instance created by the division operator. 
# -------------------------------------------------------------------

$mult = $mult/1.5;

my $div  = 1./(1.+ $mult);
my $tdiv = $div*(1. + $mult);

# -------------------------------------------------------------------
# Calculate the square root of the Tps object.
# $sqroot refers to the new Tps instance created by the sqrt method. 
# -------------------------------------------------------------------

my $sqroot  = 1. + $mult;
$sqroot  = $sqroot->sqrt();
my $tsqroot = 1. + $mult - $sqroot*$sqroot;

# -------------------------------------------------------------------
# Calculate the partial derivative of the Tps object with respect to 
# 0-variable.
# $drv refers to the new Tps instance created by the D method.
# -------------------------------------------------------------------

my $drv = $tps->D(0);
$drv->order($tps->order);

# -------------------------------------------------------------------
# Calculate the poisson bracket.
# $poisson refers to the new Tps instance created by the poisson method.
# -------------------------------------------------------------------

my $poisson = $tps->poisson(2*$tps);

# -------------------------------------------------------------------
# Get Tps coefficients and write them into the file.
# -------------------------------------------------------------------

open(TPS, ">./tps_perl.new") || die "can't create file(tps_perl)";

for($i=0; $i < $tps->size; $i++){ 
    $output = sprintf("%3d %- 12.6e %- 12.6e %- 12.6e\n", 
       $i, $mult->value($i), $tdiv->value($i), $tsqroot->value($i));
    print TPS $output;
}

close(TPS);

1;
