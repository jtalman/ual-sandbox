use lib ("$ENV{UAL_PAC}/api", "$ENV{UAL_TEAPOT}/api", "$ENV{UAL_ZLIB}/api");
 
use Pac;  
use Teapot; 
use Zlib::Tps; 

# ACCELERATOR DESCRIPTION

# Permanent part
require './local/ring.pl';

# Variable part
require './local/migrator.pl';

# ACTIONS

# Here you can call your favorite C/C++ accelerator 
# libraries ( Tracking, ZLIB(DA), Analysis, etc. )

$teapot = new Teapot::Main;
$teapot->use($ring);
$teapot->makethin();

# Make survey

require './local/survey.pl';

# Track particles

require './local/tracking.pl';

