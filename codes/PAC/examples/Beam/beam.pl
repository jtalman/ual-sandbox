# File        : samples.pl
# Description : These samples illustrate the Perl interface to library Beam
# Author      : Nikolay Malitsky 

use lib ("$ENV{UAL_PAC}/api");

use Pac::Beam;

# ********************************************************
# Initialization of particle coordinates
# ********************************************************

my $p1 = new Pac::Position;
$p1->set(1.0e-3, 0.0, 1.0e-3, 0.0, 0.0, 1.0e-5);

my $b = new Pac::Bunch(2);

for($i = 0; $i < $b->size; $i++){
    $b->position($i, ($i + 1)*$p1);
    $p2 = $b->position($i);
    print "i  = ", $i, "\n";
    print "x  = ", $p2->x,  "\n";
    print "px = ", $p2->px, "\n";
    print "y  = ", $p2->y,  "\n";
    print "py = ", $p2->py, "\n";
    print "ct = ", $p2->ct, "\n";
    print "de = ", $p2->de, "\n";
}

my $b2 = new Pac::Bunch(3);
$b->add($b2);

print "b + b2 = ", $b->size(), "\n";

