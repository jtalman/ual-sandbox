// Library     : PAC
// File        : Survey/PacSurveyDrift.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Survey/PacSurveyDrift.h"

void PacSurveyDrift::define(double length)
{
  /*
  if(length < 0.0) {
        cerr << "Error in PacSurveyDrift::define(double length) : "
         << " length(" << length << ") < 0 \n";
    exit(1);
  }
  */

 _survey.suml(length);
 _survey.z(length);


}

void PacSurveyDrift::propagate(PacSurveyData& data)
{
  PacSurveyDrift* tmp = (PacSurveyDrift *) &data;

  double length = _survey.suml();

  // x, y, z
  for(int i=0; i < 3; i++) tmp->_survey[i] += (*(tmp->_rows[i]))[2]*length;

  tmp->_survey.suml() += length;
}
