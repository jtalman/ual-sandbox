// Library     : PAC
// File        : Survey/PacSurveyData.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Survey/PacSurveyData.h"

// Private methods

void PacSurveyData::initialize()
{
  for(int i=0; i < rows(); i++) (*(_rows[i]))[i] = 1.0;
}

void PacSurveyData::define(const PacSurveyData& data)
{
  assert(rows() == data.rows() && columns() == data.columns());
  for(int i=0; i < data.rows(); i++) *(_rows[i]) = data[i];

  _survey = data._survey;
}

void PacSurveyData::define(const PacSurvey& survey)
{
  _survey = survey;

  double cosphi = cos(_survey.phi()),
         sinphi = sin(_survey.phi()),
	 costhe = cos(_survey.theta()),
	 sinthe = sin(_survey.theta()),
	 cospsi = cos(_survey.psi()),
	 sinpsi = sin(_survey.psi());

  (*_rows[0])[0] = + costhe*cospsi - sinthe*sinphi*sinpsi;
  (*_rows[0])[1] = - costhe*sinpsi - sinthe*sinphi*cospsi;
  (*_rows[0])[2] = sinthe*cosphi;

  (*_rows[1])[0] = cosphi*sinpsi;
  (*_rows[1])[1] = cosphi*cospsi;
  (*_rows[1])[2] = sinphi;

  (*_rows[2])[0] = - sinthe*cospsi - costhe*sinphi*sinpsi;
  (*_rows[2])[1] = + sinthe*sinpsi - costhe*sinphi*cospsi;
  (*_rows[2])[2] = costhe*cosphi;
}

