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

my $tps = new Zlib::Tps;

my $someOrder = $maxOrder;
$tps->order($someOrder);

# -------------------------------------------------------------------
# Set Tps data members (order & coefficients).
# -------------------------------------------------------------------

for($i = 0; $i < $tps->size; $i++) { $tps->value($i, ($i + 1.)*0.01); }

# -------------------------------------------------------------------
# Create the VTps instance, a vector of $size Tps objects.
# -------------------------------------------------------------------

my $size = $dimension;
my $vtps = new Zlib::VTps($size);

# -------------------------------------------------------------------
# Set VTps components.
# -------------------------------------------------------------------

for($i = 0; $i < $vtps->size; $i++) { $vtps->value($i, ($i + 1.)*$tps); }

# -------------------------------------------------------------------
# Copy "vtps" componets into the "copy" object. In Perl, the copy operator  
# can not be overloaded and has to be replaced by other operators.
# $copy refers to the new VTps instance created by the addition operator.
# -------------------------------------------------------------------

my $copy = $vtps + 0.0;

# -------------------------------------------------------------------
# Call the addition operators with VTps objects and numbers. 
# $sum refers to the new VTps instance created by the addition operator.
# -------------------------------------------------------------------

my $sum = $copy + $vtps + 1.0;

# -------------------------------------------------------------------
# Call the subtraction operators with VTps objects and numbers.
# $diff refers to the new VTps instance created by the subtraction 
# operator.
# -------------------------------------------------------------------

my $diff = $copy - $vtps - 2.0;

# -------------------------------------------------------------------
# Call the multiplication operator with VTps objects and numbers.
# $mult refers to the new VTps instance created by the multiplication 
# operator.
# -------------------------------------------------------------------

print "Multiplication \n";

my $mult;

print "start = ", time, "\n";
$mult = $vtps*(5*$vtps);
print "finish = ", time, "\n";

# -------------------------------------------------------------------
# Calculate the poisson bracket.
# $poisson refers to the new VTps instance created by the poisson method.
# -------------------------------------------------------------------

my $poisson = $tps->vpoisson($mult);

# -------------------------------------------------------------------
# Write VTps components into the file.
# -------------------------------------------------------------------

$mult->write("./vtps_perl.new");

