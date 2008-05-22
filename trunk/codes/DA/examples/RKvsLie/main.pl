use lib ("$ENV{UAL_ZLIB}/api", "$ENV{UAL_DA}/api");

use Zlib::Tps;
use Da::Rk::Multipole;
use Da::Lie::Multipole;

# DA Space

$dimension = 6;
$maxOrder  = 8;

$space = new Zlib::Space($dimension, $maxOrder);

# Identity

$I  = new Zlib::VTps($dimension);
$I += 1.0;

# Multipole attributes

$l = 4.00; # m

$kl  = [1.0e-4, 616.16e-4, 17.2e-4, 0.50e-4,  0.2e-4,  0.1e-4,  0.1e-4];
$ktl = [0.0,      0.20e-4, -0.5e-4, 0.25e-4, -0.2e-4, -0.25e-4, 0.8e-4];

# ****************************************************************************
print "Lie Integrator \n";
# ****************************************************************************

# Multipole as Lie Integrator, where N - max. order of exponential series 

$lie_mlt = new Da::Lie::Multipole(L => $l, N => 20 , KL  => $kl, KTL => $ktl);

# Propagate map through this element

$lie_map = $I + 0.0;

print " start  = ", time, "\n";
$lie_mlt->propagate($lie_map, {ENERGY => 8.9382796});
print " finish = ", time, "\n\n";

# I/0

$lie_map->order($maxOrder - 2);
$lie_map->write("./out/lie.new");

# ***************************************************************************
print "Runge-Kutta Integrator \n";
# ***************************************************************************

# Decrease max order of power series for Runge-Kutta integrator because 
# Lie integrator employs derivatives

$space->mltOrder($maxOrder - 2);

# Multipole as RK Integrator, where N - number of slices

$rk_mlt = new Da::Rk::Multipole(L => $l, N => 40, KL  => $kl, KTL => $ktl);

# Propagate map through this element

$rk_map = $I + 0.0;

print " start  = ", time, "\n";
$rk_mlt->propagate($rk_map, {ENERGY => 8.9382796});
print " finish = ", time, "\n\n";

# I/0

$rk_map->write("./out/rk.new");

# ****************************************************************************
print "Comparison of two results \n";
# ****************************************************************************

$dif_map = $lie_map - $rk_map;

# I/0

$dif_map->write("./out/dif.new");

$mi = 0;
$mj = 0;
$mdif = 0.0;
for($i=0; $i < $dif_map->size; $i++){
    $dif_tps = $dif_map->value($i);
    $lie_tps = $lie_map->value($i);
    for($j=0; $j < $dif_tps->size; $j++){
        $dv  = abs($dif_tps->value($j));
        $v0  = abs($lie_tps->value($j));
        if($v0) { $dv0 = $dv/$v0; }
        else    { $dv0 = 0.0; }
        if($dv0 > $mdif ){
	    $mdif = $dv0;
            $mi   = $i;
            $mj   = $j;
	}
    }
}

print " Max deviation was ", $mdif, " for coefficients (", $mi, ",", $mj, ") \n\n";
