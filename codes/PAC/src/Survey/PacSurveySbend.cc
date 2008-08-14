// Library     : PAC
// File        : Survey/PacSurveySbend.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Survey/PacSurveySbend.h"
#include <stdlib.h>

void PacSurveySbend::define(double length, double angle, double rotation)
{
  if(length < 0.0) {
    cerr << "Error in PacSurveySbend::define(double length, double angle, double rotation) : "
         << " length(" << length << ") < 0 \n";
    exit(1);
  }

  PacSurvey _tmp;

  _tmp.suml(length);
  _tmp.theta(-angle);

  if(angle == 0.0 || length  == 0.0){
    _tmp.z(length);
  }
  else{
    double radius = fabs(length/angle);
    
    _tmp.x(radius*(cos(angle) - 1.0));
    _tmp.z(radius*sin(-angle));
  } 

  PacSurveyData::define(_tmp);

  if(rotation) {
    cerr << "Error in PacSurveySbend::define(double length, double angle, double rotation) : "
         << " this version doesn't support the case when rotation != 0 \n";
    exit(1);
  }  
}  

void PacSurveySbend::propagate(PacSurveyData& data)
{
  PacSurveySbend* tmp = (PacSurveySbend* ) &data;

  // x,y,z
  for(int i=0; i < 3; i++) 
    for(int j=0; j < 3; j++)
      tmp->_survey[i] += (*(tmp->_rows[i]))[j]*_survey[j];

  // suml
  tmp->_survey.suml() += _survey.suml();

  // matrix

  double theta = - _survey.theta(); // !!! To copy Teapot algoritms
  double psi   =   _survey.psi();

  if(theta != 0.0 || psi != 0.0){

    PacSurveySbend wt;

    double costhe = cos(theta),
           sinthe = sin(theta),
	   cospsi = cos(psi),
	   sinpsi = sin(psi);

    int i;
    for(i=0; i < 3; i++){
      (*(wt._rows[i]))[0] =  (*(tmp->_rows[i]))[0]*cospsi +  (*(tmp->_rows[i]))[1]*sinpsi;
      (*(wt._rows[i]))[1] = -(*(tmp->_rows[i]))[0]*sinpsi +  (*(tmp->_rows[i]))[1]*cospsi;
      (*(wt._rows[i]))[2] =  (*(tmp->_rows[i]))[2];
    }

    double wtsi1;
    for(i=0; i < 3; i++){
      wtsi1    = (*(wt._rows[i]))[0]*costhe + (*(wt._rows[i]))[2]*sinthe;
      (*(tmp->_rows[i]))[0] = wtsi1*cospsi - (*(wt._rows[i]))[1]*sinpsi;
      (*(tmp->_rows[i]))[1] = wtsi1*sinpsi + (*(wt._rows[i]))[1]*cospsi;
      (*(tmp->_rows[i]))[2] = -(*(wt._rows[i]))[0]*sinthe + (*(wt._rows[i]))[2]*costhe;
    }
  }

  // angles
  tmp->surang();

}
