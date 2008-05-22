// Program     : Teapot
// File        : Integrator/TeapotIntegrator.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_INTEGRATOR_H
#define TEAPOT_INTEGRATOR_H

#include "Integrator/TeapotEngine.h"
#include "PAC/Beam/Position.hh"

class Teapot;

class TeapotIntegrator : public TeapotEngine<double, PAC::Position>
{

public:

  // Constructors

  TeapotIntegrator();

  // Commands

  int propagate(const PacGenElement& ge, PAC::BeamAttributes& ba, PAC::Position& p);
  int propagate(const TeapotElement& te, PAC::BeamAttributes& ba, PAC::Position& p);

  // Others

  int propagate(const TeapotElement& te, 
		PAC::Position& p, 
		PAC::Position& tmp, 
		PAC::BeamAttributes& ba, 
		double* v0byc);

  // Friend

  friend class Teapot;

protected :

  // Aperture
  int testAperture(PAC::Position& p);

  // RF  
  void passRfKick(int iKick, 
		  PAC::Position& p, 
		  PAC::Position& tmp, 
		  PAC::BeamAttributes& ba, 
		  double* v0byc);
};

#endif
