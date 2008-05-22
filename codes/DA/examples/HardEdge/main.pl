use lib ("$ENV{UAL_ZLIB}/api", "$ENV{UAL_DA}/api");

use Zlib::Tps;
use HardEdge;

# DA Space

$dimension = 6;
$maxOrder  = 5;

$space = new Zlib::Space($dimension, $maxOrder);

# Identity

$I  = new Zlib::VTps($dimension);
$I += 1.0;

# Lie integrator
# N - number of terms in the Lie transformation
# K - MAD quad coefficient multiplied by +1 (entrance) or -1 (exit)

$ff = new HardEdge("N" => 20, "K" => -4.353051/5.6575, );

# *****************************************************
print "Propagate map through this element \n";
# *****************************************************

$beam_att = {}; 
$ff_map = $I + 0.0;

print " start  = ", time, "\n";
$ff->propagate($ff_map, $beam_att);
print " finish = ", time, "\n\n";

# I/0

$ff_map->order($maxOrder - 2);
$ff_map->write("./out/ff_map.new");






