use lib ("$ENV{UAL_ZLIB}/api", "$ENV{UAL_DA}/api");

use Zlib::Tps;
use HardEdge;

# ********************************************************************
# Define the space of truncated power series: dimension and max order
# ********************************************************************

my $dimension = 6;
my $maxOrder  = 5;

my $space = new Zlib::Space($dimension, $maxOrder);

# ********************************************************************
# Create the identity map I
# ********************************************************************

my $I  = new Zlib::VTps($dimension);
$I += 1.0;

# ********************************************************************
# Create the fringe field integrator based on the Lie operator 
# ********************************************************************

# N - number of terms in the Lie transformation
# K - MAD quad coefficient multiplied by +1 (entrance) or -1 (exit)

my $ff_integrator = new HardEdge("N" => 1, "K" => -4.353051/5.6575, );

# ********************************************************************
# Propagate map through the fringe field  element;
# ********************************************************************

# array of beam attributes
my $beam_att = {}; 

# initial map
my $ff_map = $I + 0.0;

print " start  = ", time, "\n";
$ff_integrator->propagate($ff_map, $beam_att);
print " finish = ", time, "\n\n";

# ********************************************************************
# I/0
# ********************************************************************

# truncate the order of power series
$ff_map->order($maxOrder - 2);

# write power series coefficients into the specified file
$ff_map->write("./out/ff_map.new");


