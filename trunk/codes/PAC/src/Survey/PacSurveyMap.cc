// Library     : PAC
// File        : Survey/PacSurveyMap.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Survey/PacSurveyMap.h"

// PacSurvey angles from rotation matrix

void PacSurveyMap::surang()
{ 
  double arg = sqrt((*_rows[1])[0]*(*_rows[1])[0] + (*_rows[1])[1]*(*_rows[1])[1]);
  _survey.phi() = atan2((*_rows[1])[2], arg);

  if(arg > 1.0e-8){
    _survey.theta() = proxim(atan2( (*_rows[0])[2], (*_rows[2])[2] ), _survey.theta()); 
    _survey.psi()   = proxim(atan2( (*_rows[1])[0], (*_rows[1])[1] ), _survey.psi()); 
  }
  else _survey.psi() =  proxim(atan2( -(*_rows[0])[1], (*_rows[0])[0] ) - _survey.theta(), _survey.psi());

}

double PacSurveyMap::proxim(double a, double b)
{
  double c  = (b - a)/2./PI;
  int ic;
  if(c >= 0.) ic = (int)(c + 0.5);
  else        ic = (int)(c - 0.5);
  c = a + 2.*PI*ic;
  return c;
}

