use lib ("$ENV{UAL_PAC}/api/", "$ENV{UAL_SXF}/api");

use Carp;

use Pac;
use UAL::SXF::Parser;

# Make SMF

local $smf = new Pac::Smf();

# Initialize input parameters

my $latticeName  = "LHC";   # lattice name
my $elementRegex = "qxl.*"; # regular expression for selecting elements
my $attributeKey = $KL2;    # use $KL<order> or $KTL<order> - global references  
                            # to SMF attribute keys. 

my $file = "lhc.v5.0.sxf";  # input file

# #############################################################
# Read a SXF file
# #############################################################

print "read the SXF file  ", time, "\n";
my $sxf_parser = new UAL::SXF::Parser();
$sxf_parser->read($file, "$file" . ".echo");
print "end                ", time, "\n";

# #############################################################
# Execute Query 
# #############################################################

executeQuery($smf, $latticeName, $elementRegex, $attributeKey, $file . ".query");

sub executeQuery
{
  my ($smf, $latticeName, $elementRegex, $attributeKey, $fileName) = @_; 

  # Get Lattice

  my $lattice;
  for(my $it = $smf->lattices->begin(); $it != $smf->lattices->end(); $it++){
      if($it->first eq $latticeName) { $lattice = $it->second; }
  }

  if(defined $lattice) {}
  else { die "There is no lattice $latticeName \n"; }

  # Select elements and print relevant data

  open(ELEMSET, ">$fileName") || die "can't create file ($fileName)";

  my $le = 0;       # lattice element
  my $position = 0; # element position 
  my $strength = 0; # attribute value
  my $output = 0;   # element output 
  
  my ($le, $position, $strength, $output) = (0, 0, 0, 0, 0);

  for(my $i=0; $i < $lattice->size; $i++){
    $le = $lattice->element($i);    	
    if($le->genName =~ $elementRegex) {
	$strength = $le->get($attributeKey);
	$output = sprintf("%5d %14.8e %-15s %-15s %14.8e\n", 
			  $i, $position, $le->name(), $le->genName(), $strength);
	print ELEMSET $output;
    }
    $position += $le->get($L); 
  }

}

