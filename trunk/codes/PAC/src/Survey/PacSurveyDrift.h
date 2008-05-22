// Library     : PAC
// File        : Survey/PacSurveyDrift.h
// Description : The class PacSurveyDrift handles the method to propagate PacSurveyData
//               through straight elements.
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky


#ifndef SURVEY_DRIFT_H
#define SURVEY_DRIFT_H

#include "Survey/PacSurveyMap.h"

/**
The class PacSurveyDrift handles the method to propagate PacSurveyData
through straight elements.
*/

class PacSurveyDrift : public PacSurveyMap
{
public:

  // Constructors

/**@name Constructors. */
//@{
  /// constructor
  PacSurveyDrift(double length = 0.0)           { define(length);}
  /// copy constructor
  PacSurveyDrift(const PacSurveyDrift& drift)   { PacSurveyData::define(drift);}
//@}
  
  // Assignment operator
  
  ///  assignment operator
  void operator = (const PacSurveyDrift& drift) { PacSurveyData::define(drift); }

  // Access 
  /// sets the length of the element [m]
  void define(double length = 0.0);

  // Propagation

  /** Propagates the Survey Data through the element. 
      The variable data is a pointer to the PacSurveyData object.
  */
  void propagate(PacSurveyData& data);

};

#endif
