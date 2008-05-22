// Program     : Teapot
// File        : Integrator/TeapotDAIntegrator.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_DA_INTEGRATOR_H
#define TEAPOT_DA_INTEGRATOR_H

#include "Integrator/TeapotEngine.h"
#include "ZLIB/Tps/VTps.hh"

class Teapot;

class TeapotDAIntegrator : public TeapotEngine<ZLIB::Tps, ZLIB::VTps>
{

public:

  // Constructors

  TeapotDAIntegrator();

  // Commands

  int propagate(const PacGenElement& ge, PAC::BeamAttributes& ba, ZLIB::VTps& zvs);  
  int propagate(const TeapotElement& te, PAC::BeamAttributes& ba, ZLIB::VTps& zvs); 

  // Others

  int propagate(const TeapotElement& te, ZLIB::VTps& p, ZLIB::VTps& tmp, PAC::BeamAttributes& ba, double* v0byc);  

  // Friends

  friend class Teapot;

protected:

  int  testAperture(ZLIB::VTps& p);
  void passRfKick(int iKick, ZLIB::VTps& p, ZLIB::VTps& tmp, PAC::BeamAttributes& ba, double* v0byc);

};

#endif
