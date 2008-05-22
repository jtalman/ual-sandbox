// Program     : Teapot
// File        : Integrator/TeapotElemBend.cc
// Copyright   : see Copyright file
// Description : 
// Author      : Nikolay Malitsky

#include "SMF/PacElemKey.h"
#include "Integrator/TeapotElemBend.h"

void TeapotElemBend::define(double l, double angle, int ir, int element_key)
{
  erase();

  // Slices

  if(ir) _size = 5*ir;
  else   _size = 2;

  _slices = new TeapotElemSlice[_size];
  if(!_slices){
     string msg = "Error : TeapotElemBend::define(l, angle, ir) : allocation failure \n";
     PacAllocError(msg).raise();
  }

  _angle = angle;

  PacSurveyData survey;
  define(survey, l, ir, 1);

  // Strengths

  _atw[0] = 0.0;
  _atw[1] = 0.0;

  if(l){  
    double angir  = _angle;
    if(ir) angir /= 4*ir;

    _btw[0] = 2.*sin(angir/2.);

    if(element_key == pacRbendKey.key()) { _atw[1] = - sin(angir)*_angle/l; }  
    else                                 { _btw[1] =   sin(angir)*_angle/l; }
  }
  else{
    _btw[0] = 0.0;
    _btw[1] = 0.0;
  }
  // if(_angle < 0) { _btw[1] = -_btw[1]; }
    
}


void TeapotElemBend::define(PacSurveyData& survey, double l, int ir, int flag)
{
  PacSurvey survey_old = survey.survey();
  PacSurvey survey_present = survey.survey();

  PacSurveyDrift sdrift1;
  PacSurveySbend ssbend1;

  // Length

  double dl = l;
  if(_angle) { dl  = 2.0*l/_angle*tan(_angle/2.); }

  sdrift1.define(dl/2.);
  ssbend1.define(0.0, _angle);

  // Simple Bend

  if(!ir){

    sdrift1.propagate(survey);
    ssbend1.propagate(survey);
    if(flag) {
      _slices[0].define(survey_old, survey_present, survey.survey());
      survey_old = survey_present;
      survey_present = survey.survey();
    }
    sdrift1.propagate(survey);    
    if(flag) { _slices[1].define(survey_old, survey_present, survey.survey()); }    
  }

  // Complex Bend

  if(ir){

    sdrift1.define(l/2./(4.*ir)); 
    ssbend1.define(0.0, _angle/(4.*ir));

    int counter = 0;
    for(int i=0; i < ir; i++){
      for(int j=0; j < 4; j++){
	sdrift1.propagate(survey); 
	ssbend1.propagate(survey); 
	if(flag) {
	  _slices[counter++].define(survey_old, survey_present, survey.survey());
	  survey_old     = survey_present;
	  survey_present = survey.survey();
	} 
	sdrift1.propagate(survey); 
      } 
      if(flag) {
	_slices[counter++].define(survey_old, survey_present, survey.survey());
	survey_old     = survey.survey();
	survey_present = survey.survey();
      }
    }   
  }   

}    

void TeapotElemBend::initialize()
{
  _slices = 0;
  _size = 0;

  _angle = 0.0;
  for(int i=0; i <= order(); i++){
    _atw[i] = 0.0;
    _btw[i] = 0.0;
  }

  _ke1 = 0.0;
  _ke2 = 0.0;
}

void TeapotElemBend::initialize(const TeapotElemBend& teb)
{
  _size   = teb._size;
  _slices = 0;
  if(_size) {
    _slices = new TeapotElemSlice[_size];
    if(!_slices){
      string msg = "Error: TeapotElemBend::initialize(const TeapotElemBend& teb) : ";
             msg += "allocation failure \n";
      PacAllocError(msg).raise();
    }
  }  
   
  int i;
  for(i=0; i < _size; i++)
    _slices[i] = teb._slices[i];

  _angle = teb._angle;
  for(i=0; i <= order(); i++){
    _atw[i] = teb._atw[i];
    _btw[i] = teb._btw[i];
  }

  _ke1 = teb._ke1;
  _ke2 = teb._ke2;    

}

void TeapotElemBend::erase()
{
  if(_slices) delete [] _slices;
  initialize();
}
