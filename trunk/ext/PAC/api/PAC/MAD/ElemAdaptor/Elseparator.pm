package Mad::Smf::ElSeparator;

use strict;
use Carp;
use vars qw(@ISA);

use lib ("$ENV{UAL_MAD}/api/");
use Mad::Smf::Helper;

use Mad::Smf::Element;
@ISA = qw(Mad::Smf::Element);

# ***************************************************
# Public Interface
# ***************************************************

sub new
{
  my ($type, $helper) = @_;
  my $this = Mad::Smf::Element->new($helper);
  $this->{keyword} = "elseparator";
  $this->{tilt}  = $helper->{attrib_keys}->{tilt};
  $this->{keys}  = ["l", "tilt"];
  return bless $this, $type;
}

1;