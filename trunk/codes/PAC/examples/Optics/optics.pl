use lib ("$ENV{UAL_PAC}/api", "$ENV{UAL_ZLIB}/api");

use Zlib::Tps;
use Pac::Beam; 
use Pac::Optics;

$dimension = 6;
$maxOrder  = 6;

$space = new Zlib::Space($dimension, $maxOrder);

# Initialize Zlib objects

$tps = new Zlib::Tps;

$someOrder = $maxOrder;
$tps->order($someOrder);

for($i = 0; $i < $tps->size; $i++) { $tps->value($i, ($i + 1.)*0.01); }

$size = $dimension;
$vtps = new Zlib::VTps($size);

for($i = 0; $i < $vtps->size; $i++) { $vtps->value($i, ($i + 1.)*$tps); }

# ******************************************************************
print " Multiplication \n";
# ******************************************************************

print "start = ", time, "\n";
$mult = $vtps *(5*$vtps);
print "finish = ", time, "\n";

# Translate Zlib::VTps to Pac::TMap

$map = new Pac::TMap($size);
$map->daVTps($mult);


# Beam

$p = new Pac::Position;
$p->set(1.0e-3, 0.0, 1.0e-3, 0.0, 0.0, 1.0e-5);

$bunch = new Pac::Bunch(1);
$bunch->position(0, $p);

# ******************************************************************
print " Tracking \n";
# ******************************************************************

$turns = 3;
$map->propagate($bunch, $turns);


#I/0

$p = $bunch->position(0);

open(TRACKING, ">./out/perl.new") || die "can't create file(tracking)";
print TRACKING $p->x, " ", $p->px, " ", $p->y, " ", $p->py, " ", $p->ct, " ", $p->de, "\n";
close(TRACKING);


