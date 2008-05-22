
# *************************************************
# TEAPOT <----> MAD
# *************************************************

sub tpot2mad {
    my ($bunch) = @_;
    my ($position, $e0, $p0, $m0, $p, $e, $i);

    $e0 = $bunch->energy;
    $m0 = $bunch->mass;

    $p0 = $e0*$e0 - $m0*$m0;
    $p0 = sqrt($p0);
    for($i = 0; $i < $bunch->size; $i++){

        $position = $bunch->position($i);

	$p = $p0*(1.0 + $position->de);
	$e = sqrt($p*$p + $m0*$m0);
	$position->de(($e - $e0)/$p0); 

        $bunch->position($i, $position);
    }   
}

sub mad2tpot {
    my ($bunch) = @_;
    my ($position, $e0, $p0, $m0, $p, $e, $i);

    $e0 = $bunch->energy;
    $m0 = $bunch->mass;

    $p0 = $e0*$e0 - $m0*$m0;
    $p0 = sqrt($p0);

    for($i = 0; $i < $bunch->size; $i++){

	$position = $bunch->position($i);

	$e = $p0*$position->de + $e0;
	$p = sqrt(($e - $m0)*($e + $m0));
	$position->de(($p - $p0)/$p0);  

        $bunch->position($i, $position); 
    } 
}

# *************************************************
# Initial beam parameters
# *************************************************

$bunch = new Pac::Bunch(1);

$e     = $bunch->energy(5.0);
$m     = $bunch->mass(0.5110340e-3); 

$clorbit = new Pac::Position;
$clorbit->set(-1.0e-4, -2.0e-4, -3.0e-4, -4.0e-4, 0.0e-3, 0.0e-3);

$p0 = new Pac::Position;
$p0->set(-1.0e-4, -1.0e-4, -1.0e-4, -1.0e-4, 0.0e-3, -1.0e-4);


# *************************************************
print "Tracking \n";
# *************************************************

$bunch->position(0, $clorbit + 3.*$p0);
tpot2mad($bunch);

$turns = 1000;
print " start   = ", time, "\n";
$teapot->track($bunch, $turns);
print " finish  = ", time, "\n";

mad2tpot($bunch);
$pout = $bunch->position(0);
$output = sprintf("%- 15.9e %- 15.9e %- 15.9e %- 15.9e %- 15.9e %- 15.9e ", 
		  $pout->x,  $pout->px, $pout->y,  $pout->py, $pout->ct, $pout->de);
print $output, "\n";


# *************************************************
print "Making Map \n";
# *************************************************

$space = new Zlib::Space(6, 6);

$beam_att = new Pac::BeamAttributes();
$beam_att->energy(5.0);
$beam_att->mass(0.5110340e-3); 

$map = new Pac::TMap(6);

print " start   = ", time, "\n";
$teapot->map($map, $beam_att, 6);
print " finish  = ", time, "\n";

$map->write("./out/map.new");
$map->read("./out/map.old");

# *************************************************
print "Mapping \n";
# *************************************************

$bunch->position(0, $clorbit + 3.*$p0);
tpot2mad($bunch);

$turns = 1000;
print " start   = ", time, "\n";
$map->propagate($bunch, $turns);
print " finish  = ", time, "\n";

mad2tpot($bunch);
$pout = $bunch->position(0);
$output = sprintf("%- 15.9e %- 15.9e %- 15.9e %- 15.9e %- 15.9e %- 15.9e ", 
		  $pout->x,  $pout->px, $pout->y,  $pout->py, $pout->ct, $pout->de);
print $output, "\n";

1;
