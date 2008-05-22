use lib ("$ENV{UAL_ZLIB}/api", "$ENV{UAL_DA}/api");

use Zlib::Tps;
use Da::Rk::Multipole;
use Wiggler;

# DA Space

$dimension = 6;
$maxOrder  = 5;

$space = new Zlib::Space($dimension, $maxOrder);

# Identity

$I  = new Zlib::VTps($dimension);
$I += 1.0;

# Beam attributes

$energy = 5.0;          # GEV
$mass   = 0.5110340e-3; # GEV

$beam_att = {ENERGY => $energy, MASS => $mass}; 

# Drifts

$drift1  = new Da::Rk::Multipole(L => 2.13455);
$drift2  = new Da::Rk::Multipole(L => 0.95090);

# Wiggler as RK Integrator

$b0 = 1.2*0.3/$energy;  # B0/BR
$kx = 6.4;              # 1/M
$phase = -4*atan2(1,1); # 

$wiggler = new Wiggler(L => 0.196, N => 100, B0 => $b0, KX => $kx, PHASE => $phase);

# *****************************************************
print "Propagate map through the one period \n";
# *****************************************************

$w_map = $I + 0.0;

print " start  = ", time, "\n";
$wiggler->propagate($w_map, $beam_att);
$w_map->order($w_map->order());
print " finish = ", time, "\n\n";


# ***************************************************** 
print "Propagate map through the compound element \n";
# *****************************************************

$map = $I + 0.0;

print " start  = ", time, "\n";

$drift1->propagate($map, $beam_att); 
$map->order($map->order());
for($i=0; $i < 12; $i++) { $map *= $w_map; }
$drift2->propagate($map, $beam_att);

print " finish = ", time, "\n\n";

# I/0

$map->write("./out/map.new");






