// Library     : PAC
// File        : Survey/PacSurveyMap.h
// Description : The class PacSurveyMap is the abstract class that defines interfaces 
//               to all maps that propagate PacSurveyData through physical elements, 
//               such as straight element (PacSurveyDrift), bending magnet (PacSurveySbend)  
//               and others. These methods are based on MAD algorithms.
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef SURVEY_MAP_H
#define SURVEY_MAP_H

#include "Survey/PacSurveyData.h"

/**
The class PacSurveyMap is the abstract class.
It defines interfaces to all maps that propagate PacSurveyData through physical elements, 
such as straight element (PacSurveyDrift), bending magnet (PacSurveySbend)  
and others. These methods are based on MAD algorithms.
*/

class PacSurveyMap : public PacSurveyData
{
public:

  // Propagation

  /** Propagates the Survey Data through the element. 
      The variable data is a pointer to the PacSurveyData object.
  */
  virtual void propagate(PacSurveyData& data) = 0;

  /// destructor
  virtual ~PacSurveyMap(){}; 

protected:

  void surang();
  double proxim(double a, double b);

};

#endif
