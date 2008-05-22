package PAC::MAD::SMF;

use strict;
use Carp;
use vars qw(@ISA);

use lib ("$ENV{UAL_PAC}/api/");
use Pac::Smf;
@ISA = qw(Pac::Smf);

use PAC::MAD::Shell;

my $shell_;

sub new
{
    my ($type, $smf) = @_;    
    my $this = new Pac::Smf();

    $shell_ = new PAC::MAD::Shell($this);

    return bless $this, $type;
}

sub store
{
    my $this = shift;
    my %params = @_;
    my $file    = $params{file};

    $shell_->save($file);
}

sub restore
{
    my $this = shift;
    my %params = @_;
    my @files  = @{$params{files}};
    my $id = 0; if(defined $params{"id"}) {$id  = $params{"id"}; }

    $shell_->call($id, @files);
}


1;

__END__


=head1

=begin html
<h1> Class <a href="./package.html"> PAC::MAD</a>::SMF</h1>
<hr>
<h3> Extends: PAC::SMF</h3>
The SMF class adapts the Shell methods to the PAC::SMF persistence service 
interface.
<hr>
<pre><h3>Sample Script:  <a href="./SMF.txt"> SMF.pl </a> </h3> </pre>
<h3> Public Methods </h3>
<ul>
<li> <b> new() </b>
<dl>
    <dt> Constructor.
</dl>
<li> <b> restore(%parameters) </b>
<dl>
    <dt> Restore SMF data structures from MAD input files.
    <dd><i> $parameters{files} </i> - a pointer to array of MAD file names.
    <dt> Example:
    <dd> restore(files => ["file1", "file2"])
</dl>
<li> <b> store(%parameters) </b>
<dl>
    <dt> Store SMF data structures to a MAD file. 
    <dd><i>$parameters{file}</i> - a MAD file name. 
    <dt> Example:
    <dd> store(file => "file1")
</dl>
</ul> 
<hr>

=end html


