printf(STDERR "Survey   - start - Process %d\n", $myid);

open(SURVEY, ">./out/survey_mpi_". $myid . ".new") || die "can't create file(survey_mpi_new.out)";

print SURVEY "-------------------------------------------------------------------\n";
print SURVEY "    #  name       suml(thick)   suml(thin)         x               \n";
print SURVEY "-------------------------------------------------------------------\n";
  
$suml = 0.;
$survey = new Pac::SurveyData;
for($i=0; $i < $teapot->size; $i++){

     $le = $teapot->element($i);    
    
     $output = sprintf("%5d %7s %14.8e %14.8e %- 14.8e\n", $i, $le->genName(), $suml, $survey->suml, $survey->x);
     print SURVEY $output;

     $teapot->survey($survey, $i, $i+1);
     $suml += $le->get($L); 
}

close(SURVEY);

printf(STDERR "Survey   - stop  - Process %d\n", $myid);

1;
