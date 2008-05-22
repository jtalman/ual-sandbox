use lib ("$ENV{UAL_PAC}/api");
use Pac::Smf;

$smf1 = new Pac::Smf();

print "\nElement keys: \n\n";

$i = 0;
for($it = $smf1->elemKeys->begin(); $it != $smf1->elemKeys->end(); $it++){
    print $i++, " ", $it->first, " ", $it->second->name, "\n";
}

$it = $smf1->elemKeys->find(3);
if( $it != $smf1->elemKeys->end()) {print "\nWe have found ", $it->second->name, "!\n";}

print "\nElement bucket keys: \n\n";

$i = 0;
for($it = $smf1->bucketKeys->begin(); $it != $smf1->bucketKeys->end(); $it++){
    print $i++, " ", $it->first, " ", $it->second->name, "(";
    for($a = 0; $a < $it->second->size; $a++) {
	print " ", $it->second->attribKey($a)->name;
    }
    print " ) ", $it->second->order, "\n";
}

$it = $smf1->bucketKeys->find(3);
if( $it != $smf1->bucketKeys->end()) {print "\nWe have found ", $it->second->name, "!\n";}


print "\nGeneric Elements: \n\n";

$i = 0;
for($it = $smf1->elements->begin(); $it != $smf1->elements->end(); $it++){
    print $i++, " ", $it->first, "\n";
}

$it = $smf1->elements->find("x");
if( $it != $smf1->elements->end()) {print "\nWe have found ", $it->first, "!\n";}

print "\nLines: \n\n";

$i = 0;
for($it = $smf1->lines->begin(); $it != $smf1->lines->end(); $it++){
    print $i++, " ", $it->first, "\n";
}

$it = $smf1->lines->find("cell10");
if( $it != $smf1->lines->end()) {print "\nWe have found ", $it->first, "!\n";}

print "\nLattices: \n\n";

$i = 0;
for($it = $smf1->lattices->begin(); $it != $smf1->lattices->end(); $it++){
    print $i++, " ", $it->first, "\n";
}

$it = $smf1->lattices->find("west");
if( $it != $smf1->lattices->end()) {print "\nWe have found ", $it->first, "!\n";}

1;
