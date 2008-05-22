use lib ("$ENV{UAL_PAC}/api");
use Pac::Smf;

$smf2 = new Pac::Smf();

@elParts = (' Front:', ' Body:', ' End:');

for($it = $smf2->bucketKeys->begin(); $it != $smf2->bucketKeys->end(); $it++){
  $bucketKeys[$it->first] = $it->second;
}

$il = 0;
for($it = $smf2->lattices->begin(); $it != $smf2->lattices->end(); $it++){
    print "\n\nLattices:  ", $il++, " ", $it->first, "\n";

print "\nElements: \n\n";

$position = 0; 
for($i = 0; $i < $it->second->size; $i++){
    $el = $it->second->element($i);
    $length = $el->get($L);
    print $i, " ", $position + $length/2., " ", $el->genName; 
    print " ", $smf2->elemKeys->find($el->key)->second->name;

    for($ip = 0; $ip < 3; $ip++){
	$part = $el->getPart($ip);
	if($part){
	    print $elParts[$ip];
	    for($ib = $part->begin(); $ib != $part->end(); $ib++){

		$size = $bucketKeys[$ib->first]->size;
		$order = $ib->second->size/$size;

		for($ia = 0; $ia != $size; $ia++){
		    for($io = 0; $io < $order; $io++){
			$value = $ib->second->value($io*$size + $ia);
			if($value){
			    print " ", $bucketKeys[$ib->first]->attribKey($ia)->name;
			    if($bucketKeys[$ib->first]->order) { print $io; }
			    print " = ", $value;
			}
		    } # order
		} # attributes
	    } # buckets
	}
    } # parts

    print "\n";
    $position += $length;

} # elements

} # lattices
		    

1;
