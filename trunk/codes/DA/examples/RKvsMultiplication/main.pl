use lib ("$ENV{UAL_ZLIB}/api", "$ENV{UAL_DA}/api");

use Zlib::Tps;
use Da::Rk::Multipole;

# DA Space

$dimension = 6;
$maxOrder  = 6;

$space = new Zlib::Space($dimension, $maxOrder);

# Identity

$I  = new Zlib::VTps($dimension);
$I += 1.0;

# ***************************************************************************
print "Initialize Runge-Kutta integrators \n";
# ***************************************************************************

# Multipole as RK Integrator, where N - number of slices

$rk1_mlt = new Da::Rk::Multipole(L => 4./10000., N => 5, KL  => [0, 616.16]);
$rk2_mlt = new Da::Rk::Multipole(L => 4., N => 5);

# ***************************************************************************
print "Propagate map(rk_12_map) through elements 1 & 2 :  \n";
# ***************************************************************************

$rk_12_map = $I + 0.0;

print " start  = ", time, "\n";
$rk1_mlt->propagate($rk_12_map, {ENERGY => 8.9382796});
$rk2_mlt->propagate($rk_12_map, {ENERGY => 8.9382796});
print " finish = ", time, "\n\n";

$rk_12_map->write("./out/rk_12.new");

# ***************************************************************************
print "Propagate map(rk_21_map) through elements 2 & 1 :  \n";
# ***************************************************************************

$rk_21_map = $I + 0.0;

print " start  = ", time, "\n";
$rk2_mlt->propagate($rk_21_map, {ENERGY => 8.9382796});
$rk1_mlt->propagate($rk_21_map, {ENERGY => 8.9382796});
print " finish = ", time, "\n\n";

$rk_21_map->write("./out/rk_21.new");

# ***************************************************************************
print "Propagate map(rk_1_map) through element 1 :  \n";
# ***************************************************************************

$rk_1_map = $I + 0.0;

print " start  = ", time, "\n";
$rk1_mlt->propagate($rk_1_map, {ENERGY => 8.9382796});
print " finish = ", time, "\n\n";

# ***************************************************************************
print "Propagate map(rk_2_map) through element 2 :  \n";
# ***************************************************************************

$rk_2_map = $I + 0.0;

print " start  = ", time, "\n";
$rk2_mlt->propagate($rk_2_map, {ENERGY => 8.9382796});
print " finish = ", time, "\n\n";

# ***************************************************************************
print "m_12_map = rk_1_map*rk_2_map == rk_12_map :  \n";
# ***************************************************************************

# It is very important ! You need to adjust Tps orders before
# multiplication. 

$rk_1_map->order($rk_1_map->order());
$rk_2_map->order($rk_2_map->order());

print " start  = ", time, "\n";
$m_12_map  = $rk_1_map*$rk_2_map;
print " finish = ", time, "\n\n";

$m_12_map->write("./out/m_12.new");


# ***************************************************************************
print "m_21_map = rk_2_map*rk_1_map == rk_21_map :  \n";
# ***************************************************************************

print " start  = ", time, "\n";
$m_21_map  = $rk_2_map*$rk_1_map;
print " finish = ", time, "\n\n";

$m_21_map->write("./out/m_21.new");


