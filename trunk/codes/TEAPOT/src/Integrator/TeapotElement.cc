// Library     : Teapot
// File        : Integrator/TeapotElement.cc
// Copyright   : see Copyright file
// Description : 
// Author      : Nikolay Malitsky

#include "SMF/PacElemLength.h"
#include "SMF/PacElemBend.h"
#include "SMF/PacElemRotation.h"
#include "SMF/PacElemComplexity.h"

#include "Integrator/TeapotElement.h"


// Operators

TeapotElement& TeapotElement::operator=(const PacGenElement& e)
{
  PacLattElement::operator=(e);
  erase();
  initialize();
  return *this;
}

TeapotElement& TeapotElement::operator=(const PacLattElement& e)
{
  PacLattElement::operator=(e);
  erase();
  initialize();
  return *this;
}

void TeapotElement::propagate(PacSurveyData& survey)
{
  if(!_bend) {
    PacSurveyDrift sdrift;
    sdrift.define(_l);
    sdrift.propagate(survey); 
    return;
  }

  _bend->propagate(survey, _l, _ir);

}


// Private methods

void TeapotElement::erase()
{
  if(_bend) delete  _bend;
  if(_rotation) delete  _rotation;
}

void TeapotElement::initialize()
{
  _l  = 0.0;
  _ir = 0;

  int bendKey = 0;
  double tilt = 0.0, angle = 0.0, e1 = 0.0, e2 = 0.0;

  PacElemLength *length = 0;
  PacElemBend *bend = 0;
  PacElemRotation *rot = 0;
  PacElemComplexity * complexity = 0;

  PacElemAttributes* attributes;

  // Front

  attributes = getFront();

  if(attributes){
   for(PacElemAttributes::iterator it = attributes->begin(); it != attributes->end(); it++)
     switch((*it).key()){
     case PAC_BEND:
       bend = (PacElemBend*) &(*it);
       e1 = bend->angle();
       if(e1) bendKey += 1;
       break;
     default:
       break;
     }
  }

  // Body

  attributes = getBody();

  if(attributes){
   for(PacElemAttributes::iterator it = attributes->begin(); it != attributes->end(); it++)
     switch((*it).key()){
     case PAC_LENGTH:
       length = (PacElemLength*) &(*it);
       _l = length->l();
       break;   
     case PAC_BEND:
       bend = (PacElemBend*) &(*it);
       angle = bend->angle();
       if(angle) bendKey += 1;
       break;
     case PAC_ROTATION:
       rot = (PacElemRotation*) &(*it);
       tilt = rot->tilt();
       break;
     case PAC_COMPLEXITY:
       complexity = (PacElemComplexity*) &(*it);
       _ir = (int) complexity->n();
       break;
     default:
       break;
    } 
  }

  // End

  attributes = getEnd();

  if(attributes){
   for(PacElemAttributes::iterator it = attributes->begin(); it != attributes->end(); it++)
     switch((*it).key()){
     case PAC_BEND:
       bend = (PacElemBend*) &(*it);
       e2 = bend->angle();
       if(e2) bendKey += 1;
       break;
     default:
       break;
     }
  }

  // Bend

  _bend = 0;

  if(bendKey) {
    _bend = new TeapotElemBend();
    if(!_bend){
      string msg = "Error : TeapotElement::initialize : allocation failure \n";
      PacAllocError(msg).raise();
    }
    _bend->define(_l, angle, _ir, PacLattElement::key());
     double rho = _l/angle;
     if(_l){
       _bend->ke1(-sin(e1)/cos(e1)/rho);
       _bend->ke2(-sin(e2)/cos(e2)/rho);
     }
     else{
       _bend->ke1(0.0);
       _bend->ke2(0.0);
     }       
  }

  // Rotation

  _rotation = 0;

  if(tilt) {
    _rotation = new TeapotElemRotation();
    if(!_rotation){
      string msg = "Error : TeapotElement::initialize : allocation failure \n";
      PacAllocError(msg).raise();
    }

    _rotation->define(tilt);
  }
}




