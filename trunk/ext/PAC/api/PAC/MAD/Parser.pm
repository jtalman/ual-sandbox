package PAC::MAD::Parser;

use strict;
use Carp;

use File::Copy;
use File::Basename;

sub new
{
 my $type = shift;
 my $this = {};
 return bless $this, $type;
}

sub translate
{
  my $this   = shift;
  my %params = @_;
  my $files  = $params{files};
  my $script = $params{script};
  my $id     = $params{id};

  # my $mad    = "parser.mad";
  # my $tmp    = "parser.tmp";

  my $basename = basename($files->[0]);

  my $mad = $basename . "parser_" . $id . ".mad";
  my $tmp = $basename . "parser_" . $id . ".tmp";


  
  $this->merge($files, $mad);  
  $this->simplify($mad, $tmp);
  $this->parse($tmp, $script);

  unlink $mad;
  unlink $tmp;

}

sub merge
{
  my ($this, $files, $mad) = @_;
  open(FH, ">$mad");
  foreach(@$files){ copy($_, \*FH); }
  close(FH);
}


sub simplify
{
  my ($this, $mad, $tmp) = @_;
  open(FH_MAD, $mad) or die "Can't access $mad file\n";

  open(FH_TMP, ">$tmp");

  my ($line, @lines);
  while (defined($line = <FH_MAD>)){
     @lines    = split /(\!)/, $line;          # separate comments
     $lines[0] =~ tr/A-Z/a-z/;                 # canonicalize to lower case
     $lines[0] =~ s/&.*\n/ /;                  # concatenate two lines
     $lines[0] =~ s/([a-z][a-z0-9]*)\./$1\_/g; # replace \. by \_ in names
     foreach (@lines) { printf FH_TMP $_ };
  }
  close(FH_MAD); 
  close(FH_TMP); 
}

sub parse
{
  my ($this, $tmp, $script) = @_;
  system("mad2smf < $tmp > $script") == 0
	or die "system failed: $?" ;
}

1;

__END__

=head1

=begin html
<h1> Class <a href="./package.html"> PAC::MAD</a>::Parser</h1>
<hr>
<h3> Extends: </h3>
The Parser class translates MAD input files with a lattice description 
to a Perl script. This script initializes the SMF data structure via the
<a href="./Shell.html"> PAC::MAD::Shell</a> interface.
<hr>
<pre><h3>Sample Script:  <a href="./Parser.txt"> Parser.pl </a> </h3></pre>
<h3> Public Methods </h3>
<ul>
<li> <b> new() </b>
<dl>
    <dt> Constructor.
</dl>
<li> <b> translate($files, $script) </b>
<dl>
    <dt>Translates  MAD input files to a Perl script.
    <dd><i>files</i> - a reference to an array of MAD input file names (e.g. ["f1", "f2"]). 
    <dd><i>script</i> - a Perl script name.
</dl>
</ul> 
<hr>

=end html
