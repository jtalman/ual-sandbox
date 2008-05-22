// Library     : PAC
// File        : Survey/PacSurveySbend.h
// Description : The class PacSurveySbend handles the method to propagate PacSurveyData
//               through bending magnets.
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef SURVEY_SBEND_H
#define SURVEY_SBEND_H

#include "Survey/PacSurveyMap.h"

/**
The class PacSurveySbend handles the method to propagate PacSurveyData
through bending magnets.
*/

class PacSurveySbend : public PacSurveyMap
{
public:

  // Constructors

/**@name Constructors. */
//@{
  /// copy constructor
  PacSurveySbend(const PacSurveySbend& bend)      {PacSurveyData::define(bend); }
  /** constructor.
      Sets the length of the element, its bending angle, and its angle of rotation.
  */
  PacSurveySbend(double length=0.0, double angle=0.0, double rotation=0.0) { define(length, angle, rotation);}
//@}
  // Assignment operator
  /// assignment operator 
  void operator = (const PacSurveySbend& bend) {PacSurveyData::define(bend); }

  // Access

  /// Sets the length of the element, its bending angle, and its angle of rotation.
  void define(double length=0.0, double angle=0.0, double rotation=0.0);

  // Propagation

  /** Propagates the Survey Data through the element. 
      The variable data is a pointer to the PacSurveyData object.
  */
  void propagate(PacSurveyData& data);

};

#endif
