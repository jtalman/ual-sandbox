# File        : samples.pl
# Description : These samples illustrate the Perl interface to library
#               Survey.
# Author      : Nikolay Malitsky 

use lib ("$ENV{UAL_PAC}/api");

use Pac::Survey;

$length   = 2.0;   # [m]
$angle    = 0.1;   # [rad]
$rotation = 0.0;   # [rad]

# ********************************************************
# Sample 1. Defining a survey map for straight elements
# ********************************************************

$drift = new Pac::SurveyDrift($length);

# ********************************************************
# Sample 2. Defining a survey map for bending magnets
# ********************************************************

$bend = new Pac::SurveySbend($length, $angle, $rotation);

# ********************************************************
# Sample 3. Propagation of survey
# ********************************************************

$survey = new Pac::SurveyData;
$bend->propagate($survey);

print "x = ", $survey->x, "\n";


