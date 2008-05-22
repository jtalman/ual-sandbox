// Library     : Teapot
// File        : Integrator/TeapotElement.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_ELEMENT_H
#define TEAPOT_ELEMENT_H

#include "SMF/PacLattElement.h"

#include "Main/TeapotDef.h"
#include "Integrator/TeapotElemBend.h"
#include "Integrator/TeapotElemRotation.h"

class TeapotElement : public PacLattElement
{
public:

  // Constructors & copy operator

  TeapotElement() : PacLattElement() {initialize();}
  TeapotElement(const PacGenElement& e) : PacLattElement(e) { initialize(); }
  TeapotElement(const PacLattElement& e): PacLattElement(e) { initialize(); }
 ~TeapotElement() {erase(); }

  TeapotElement& operator = (const PacGenElement& e);
  TeapotElement& operator = (const PacLattElement& e);

  // Access

  double l() const { return _l; }
  int ir() const { return _ir; }

  TeapotElemBend* bend() const { return _bend; }  
  TeapotElemRotation* rotation() const { return _rotation; }

  // Commands

  void propagate(PacSurveyData& survey);

protected:

  // Data

  double _l;
  int _ir;
  TeapotElemBend* _bend;
  TeapotElemRotation* _rotation;

private:

  void initialize();
  void erase();

};
  
#endif
