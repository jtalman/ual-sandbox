// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DipoleData.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "SMF/PacElemBend.h"
#include "SMF/PacElemAttributes.h"
#include "ETEAPOT/Integrator/DipoleData.hh"

ETEAPOT::DipoleData::DipoleData()
{
  initialize();
}

ETEAPOT::DipoleData::DipoleData(const DipoleData& data)
{
  copy(data);
}

ETEAPOT::DipoleData::~DipoleData()
{
}

const ETEAPOT::DipoleData& ETEAPOT::DipoleData::operator=(const ETEAPOT::DipoleData& data)
{
  copy(data);
  return *this;
}

void ETEAPOT::DipoleData::setLatticeElement(const PacLattElement& e)
{
  // length
  m_l = e.getLength();

  // ir
  m_ir = e.getN();

  // bend angle and strengths

  // double angle = 0.0;

  // Body attributes
  PacElemAttributes* attributes = e.getBody();
  PacElemBend* bend;

  if(attributes){
    for(PacElemAttributes::iterator it = attributes->begin(); it != attributes->end(); it++){
      switch((*it).key()){
      case PAC_BEND:
	bend = (PacElemBend*) &(*it);
	m_angle = bend->angle();
	break;
      default:
	break;
      }
    }
  }
  setBendStrengths(e.getType());

  // std::cerr << e.genElement().name() << ", l = " << m_l <<  ", angle = " << m_angle << ", ir = " << m_ir << "\n";

  // slices
  setSlices(m_l, m_angle, (int) m_ir);
  
}

void ETEAPOT::DipoleData::setBendStrengths(const std::string& etype)
{
  // m_angle = angle;

  // Strengths

  m_atw00 = 0.0;
  m_atw01 = 0.0;
  m_btw00 = 0.0;
  m_btw01 = 0.0;

  if(m_l){  
    double angir  = m_angle;
    if(m_ir) angir /= 4*m_ir;

    m_btw00 = 2.*sin(angir/2.);
    m_btw01 = sin(angir)*m_angle/m_l;
  }

  if(etype == "Rbend"){
    m_atw01 = - m_btw01;
    m_btw01 = 0.0;
  }

  // if(angle < 0) { m_btw01 = -m_btw01; }
}


void ETEAPOT::DipoleData::setSlices(double l, 
				   double angle, 
				   int ir)
{
  // Clean old data
  m_slices.clear();

  // Allocate memory
  int nslices = 2;
  if(ir) nslices = 5*ir;
  m_slices.resize(nslices);

  // Propagate survey and calculate slice parameters
  PacSurveyData survey;
  setSlices(survey, l, angle, ir, 1);
    
}

void ETEAPOT::DipoleData::setSlices(PacSurveyData& survey, 
				   double l, 
				   double angle, 
				   int ir, 
				   int flag)
{
  PacSurvey survey_old = survey.survey();
  PacSurvey survey_present = survey.survey();

  PacSurveyDrift sdrift1;
  PacSurveySbend ssbend1;

  // Length

  double dl = l;
  if(angle) { dl  = 2.0*l/angle*tan(angle/2.); }

  sdrift1.define(dl/2.);
  ssbend1.define(0.0, angle);

  // Simple Bend

  if(!ir){

    sdrift1.propagate(survey);
    ssbend1.propagate(survey);
    if(flag) {
      m_slices[0].define(survey_old, survey_present, survey.survey());
      survey_old = survey_present;
      survey_present = survey.survey();
    }
    sdrift1.propagate(survey);    
    if(flag) { m_slices[1].define(survey_old, survey_present, survey.survey()); }    
  }

  // Complex Bend

  if(ir){

    sdrift1.define(l/2./(4.*ir)); 
    ssbend1.define(0.0, angle/(4.*ir));

    int counter = 0;
    for(int i=0; i < ir; i++){
      for(int j=0; j < 4; j++){
	sdrift1.propagate(survey); 
	ssbend1.propagate(survey); 
	if(flag) {
	  m_slices[counter++].define(survey_old, survey_present, survey.survey());
	  survey_old     = survey_present;
	  survey_present = survey.survey();
	} 
	sdrift1.propagate(survey); 
      } 
      if(flag) {
	m_slices[counter++].define(survey_old, survey_present, survey.survey());
	survey_old     = survey.survey();
	survey_present = survey.survey();
      }
    }   
  }   

}  

void ETEAPOT::DipoleData::initialize()
{
  // length

  m_l = 0.0;

  // ir

  m_ir = 0.0;

  // bend angle and strengths

  m_angle = 0.0;

  // Strengths

  m_atw00 = 0.0;
  m_atw01 = 0.0;
  m_btw00 = 0.0;
  m_btw01 = 0.0;

  m_m = 0.0;


}  

void ETEAPOT::DipoleData::copy(const DipoleData& data)
{
  // length

  m_l = data.m_l;

  // ir

  m_ir = data.m_ir;

  // bend angle and strengths

  m_angle = data.m_angle;

  // Strengths

  m_atw00 = data.m_atw00;
  m_atw01 = data.m_atw01;
  m_btw00 = data.m_btw00;
  m_btw01 = data.m_btw01;

  m_slices = data.m_slices;

  m_m = data.m_m;

}  

