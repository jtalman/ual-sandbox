#
#   Fermilab 8.9 Gev/c TOY Recycler lattice:
#
#
#   INITIAL LATTICE: D. Johnson  October 10, 1995
#
#   CURRENT VERSION:  DEC 6, 1996  ~dej/recycler/ring/toy/toy.lat



use lib ("$ENV{UAL_PAC}/api", "$ENV{UAL_ZLIB}/api");

use Zlib::Tps;
use Pac::Beam; 
use Pac::Optics;
use Pac::Smf;



$smf = new Pac::Smf;

#
#   Define energy and brho
#

$ke   = 8.0;
$m0   = 0.93826;
$p0   = sqrt($ke*(2*$m0 + $ke));
$brho = $p0/0.2997925;           # T-m

#
#  Define cell lengths for arc,dispersion suppressor and
#  straight section
#

$larccell = 17.288191638;

#
# Define the arc magnet magnetic length as 4.064 m (160 inches)
# Define the end pack to be 2.5 inches so the full length of the
# magnet is 165 inches
#

$lbarcmag =  4.2672;                  # (168") 4.064   !old value 3.645
$lbarcend =  0.0889;                  # (3.5") 0.0635  !old value 0.127
$lbarcphy =  $lbarcmag + 2*$lbarcend; 

#
#  Define the bend angle, theta,  for the arc gradient
#  magents assuming there are 300 equivalent gradient
#  magnets
#

$pi = 3.1415926535898;

$ndipoles = 300;
$theta = 2*$pi/$ndipoles;

$dir = +1;  # for pbars
$b0  = $dir*$theta*$brho/$lbarcmag;

#
# Define the bend angle for both dipoles
#

$barcang = $b0*$lbarcmag/$brho;

# Define arc gradient magnet gradients for 4.2672 m magnets
#   psix=86.8deg/cell  psiy=79.3deg/cell
#

$barck1f =  1.185335E-02;
$barck1d = -1.135072E-02;

$gfarc = $barck1f*$brho;
$gdarc = $barck1d*$brho;

#
# Sextupole in the arc combined function
#

$barck2f =  5.538112E-03;
$barck2d = -9.169916E-03;

$sfbarc = $barck2f*$brho;
$sdbarc = $barck2d*$brho;


# Dipoles

$smf->elements->declare($Sbend, arcf, arcd);

$arcf->set($lbarcmag*$L, $barcang*$ANGLE, 4.0*$N, ($barck1f*$lbarcmag)*$KL1, ($barck2f*$lbarcmag/2.)*$KL2);
$arcf->front->set(($barcang/2.)*$ANGLE);
$arcf->end->set(($barcang/2.)*$ANGLE);

$arcd->set($lbarcmag*$L, $barcang*$ANGLE, 4.*$N, ($barck1d*$lbarcmag)*$KL1, ($barck2d*$lbarcmag/2.)*$KL2);
$arcd->front->set(($barcang/2.)*$ANGLE);
$arcd->end->set(($barcang/2.)*$ANGLE);

#
# Define Physical magnet
#

$smf->elements->declare($Drift, arcend);
$smf->lines->declare(barcf, barcd);

$arcend->set($lbarcend*$L);

$barcf->set($arcend, $arcf, $arcend);
$barcd->set($arcend, $arcd, $arcend);

# Drifts

$smf->elements->declare($Drift, dbb, dcell);

#
#  Define  drifts between half cells for arc cells, dispersion suppressors
#  and quads
#

$ldbb = 0.5;
$dbb->set($ldbb*$L);

#
# Define length of long drift space between F and D arc cell gradient magnets
#

$dcell->set(($larccell - 2*$lbarcphy - 2*$ldbb)*$L);

# Markers

$smf->elements->declare($Marker, mf, md);

# Lines

$smf->lines->declare(hcfd, hcdf, cellff, ring);

# Define halfcells

$hcfd->set($mf, $dbb, $barcf, $dcell, $barcd, $dbb, $md);
$hcdf->set($md, $dbb, $barcd, $dcell, $barcf, $dbb, $mf);

$cellff->set($hcfd, $hcdf);

$ring->set(75*$cellff);

# Lattices

$smf->lattices->declare(toy);
$toy->set($cellff);

print " toy.pl - STOP.\n";

1;
