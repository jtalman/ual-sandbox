package Pac::Smf;

use lib ("$ENV{UAL_PAC}/api/blib/$ENV{UAL_ARCH}/arch",
         "$ENV{UAL_PAC}/api/blib/$ENV{UAL_ARCH}/lib");

use strict;
use Carp;
use vars qw($VERSION @ISA @EXPORT $AUTOLOAD);

require Exporter;
require DynaLoader;
require AutoLoader;

@ISA = qw(Exporter DynaLoader);
@EXPORT = qw();
$VERSION = '1.0';

bootstrap Pac::Smf $VERSION;

package Pac::Smf;

my $smf_flag = 0;

sub new
{
  my $type = shift;
  my $this = Pac::Smf::create();
  bless $this, $type;
  if($smf_flag == 0) { $this->initialize(); $smf_flag = 1; }
  return $this;
}

package Pac::Lattice;

sub indexes
{
   my ($self, $regex) = @_;
   my (@v, $i);
   for($i = 0; $i < $self->size(); $i++){
	if($self->element($i)->genName =~ $regex) { push @v, $i;}
   }
   return @v;   
}

package Pac::ElemKeyIterator;

use overload
	"++" => \&add,
	"!=" => \&ne;

package Pac::ElemBucketKeyIterator;

use overload
	"++" => \&add,
	"!=" => \&ne;

package Pac::ElemAttribKey;

use overload
	"*" => \&multiply;

package Pac::ElemAttribIterator;

use overload
	"++" => \&add,
	"!=" => \&ne;

package Pac::GenElement;

use overload
	"*" => \&multiply;

package Pac::GenElemIterator;

use overload
	"++" => \&add,
	"!=" => \&ne;

package Pac::Line;

use overload
	"*" => \&multiply;

package Pac::LineIterator;

use overload
	"++" => \&add,
	"!=" => \&ne;

package Pac::LatticeIterator;

use overload
	"++" => \&add,
	"!=" => \&ne;

package Pac::LattElement;

package Pac::Smf;




1;
__END__
