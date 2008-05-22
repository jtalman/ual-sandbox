// Library     : PAC
// File        : Survey/PacSurveyData.h
// Description : The class PacSurveyData contains two representations of global 
//               coordinates in  vector (PacSurvey) and matrix (PacMatrix) forms.  
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef SURVEY_DATA_H
#define SURVEY_DATA_H

#include "UAL/Common/Object.hh"
#include "Templates/PacMatrix.h"
#include "Survey/PacSurvey.h"

/**
 The class PacSurveyData contains two representations of global 
 coordinates in  vector (PacSurvey) and matrix (PacMatrix) forms.
*/


class PacSurveyData : protected PacMatrix<double>, public UAL::Object
{
public:

  // Constructors
/**@name Constructors. */
//@{
  /// constructor
  PacSurveyData()                       : PacMatrix<double>(3,3, 0.0)  {initialize();}
  /// copy constructor
  PacSurveyData(const PacSurveyData& data) : PacMatrix<double>(data)   {_survey = data._survey;}
  /// constructor intializes PacSurveyData from PacSurvey object's data
  PacSurveyData(const PacSurvey& survey)   : PacMatrix<double>(3,3)    {define(survey);}
//@}

  // Assignment  operators

/**@name Assignment  operators. */
//@{
  /// from PacSurveyData object's data
  void operator  = (const PacSurveyData& data)                         {define(data);}
  /// from PacSurvey object's data
  void operator  = (const PacSurvey& survey)                           {define(survey);}
//@}

  // Access
  /// access
  const PacSurvey& survey() const                                      {return _survey;}

protected:

  PacSurvey _survey;

  void initialize();
  void define(const PacSurvey& survey);
  void define(const PacSurveyData& data);

};

#endif
