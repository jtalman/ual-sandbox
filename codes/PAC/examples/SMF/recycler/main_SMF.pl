use lib ("$ENV{UAL_PAC}/api");
use Pac::Smf;

# ACCELERATOR DESCRIPTION

# Regular (MAD) elements, lines, lattices

require 'toy.pl';

# ACTIONS

# Here you can call your favorite C/C++ accelerator 
# libraries ( Tracking, DA, Analysis, etc. )

# DEVELOPER'S CORNER

# We employ the STL style to access SMF collections, 
# associative arrays of keys (first) and pointers to 
# objects (second).

# Here is a simple output. 
# Certainly, you can build your favorite format.

print "\n************************************\n";
print "\nBuild a simple output \n";
print "\n************************************\n";

require '../util/output.pl';

print "\n************************************\n";
print "\nCheck all SMF collections \n";
print "\n************************************\n";

require '../util/collections.pl';

