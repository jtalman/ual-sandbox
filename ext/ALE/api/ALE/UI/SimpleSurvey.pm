package ALE::UI::SimpleSurvey;

use strict;
use Carp;

my $L_ = 0;

sub new
{
    my ($type, $shell) = @_;
    my $this = {};

    $shell->map->attribKeyFromString("l", \$L_);

    return bless $this, $type;
}

sub print
{
    my ($this, $file, $regex, $shell) = @_;

    my $lattice = $shell->{lattice};
    my $code = $shell->{code};

    
    open(SURVEY, ">$file") || die "can't create file(survey)";

    my @columns = ("#", "name", "suml(thick)", "suml(thin)", "x", "y", "z", "theta", "phi", "psi");    
    print SURVEY "------------------------------------------------------------";
    print SURVEY "------------------------------------------------------------\n"; 
    # print SURVEY  "    #   name            suml(thick)    suml(thin)         x               \n";
    my $output = sprintf("%-5s %-10s   %-15s   %-15s %- 10s  %- 10s  %- 10s  %- 10s  %- 10s  %- 10s\n", 
	$columns[0],  $columns[1], $columns[2], $columns[3],  $columns[4],
	$columns[5],  $columns[6], $columns[7], $columns[8], $columns[9]);
    print SURVEY $output;
    print SURVEY "------------------------------------------------------------";
    print SURVEY "------------------------------------------------------------\n"; 

    my $survey = new Pac::SurveyData; 
    my ($i, $le, $suml, $output) = (0, 0, 0, 0);

    for($i=0; $i < $lattice->size; $i++){
	$le = $lattice->element($i);    
    
	if($le->genName =~ $regex) {
	    $output = sprintf("%5d %-10s %15.7e %15.7e %- 10.4e %- 10.4e %- 10.4e %- 10.4e %- 10.4e %- 10.4e\n", 
			      $i, $le->genName(), $suml, $survey->suml, 
			      $survey->x, $survey->y, $survey->z, 
			      $survey->theta, $survey->phi, $survey->psi);
	    print SURVEY $output;
	}

	$code->survey($survey, $i, $i+1);
	$suml += $le->get($L_); 
    }

    my ($j, $end) = (0, "End");

    $output = sprintf("%5d %-10s %15.7e %15.7e %- 10.4e %- 10.4e %- 10.4e %- 10.4e %- 10.4e %- 10.4e\n", 
		      $j, $end, $suml, $survey->suml, 
		      $survey->x, $survey->y, $survey->z, 
		      $survey->theta, $survey->phi, $survey->psi);
    print SURVEY $output;

    close(SURVEY);
}

1;

__END__


=head1

=begin html
<h1> Class <a href="./package.html"> ALE::UI</a>::SimpleSurvey </h1>
<hr>
<h3> Extends: </h3>
The SimpleSurvey class calculates an accelerator geometry (survey) and writes
the results to a file. 
<hr>
<h3> Public Methods </h3>
<ul>
<li> <b> new($shell) </b>
<dl>
    <dt> Constructor.
    <dd><i>shell</i> - a pointer to a PAC::MAD::Shell instance.
</dl>
<li> <b> print($file, $pattern, $shell) </b>
<dl>
    <dt> Calculates an accelerator survey for selected elements and 
	writes a simple output.
    <dd><i>file</i>    - a output file.
    <dd><i>pattern</i> - a regular expression for selecting elements. 
    <dd><i>shell</i>   - a pointer to a ALE::UI::Shell instance.  
</dl>
</ul> 
<hr>

=end html



