// Program     : Teapot
// File        : Integrator/TeapotDAIntegrator.cc
// Copyright   : see Copyright file
// Description : 
// Author      : Nikolay Malitsky

#include "Integrator/TeapotDAIntegrator.h"

TeapotDAIntegrator::TeapotDAIntegrator()
{
}

int TeapotDAIntegrator::propagate(const PacGenElement& ge, PAC::BeamAttributes& ba, ZLIB::VTps& p)
{
  return TeapotEngine<ZLIB::Tps, ZLIB::VTps>::propagate(ge, ba, p);
}

int TeapotDAIntegrator::propagate(const TeapotElement& te, PAC::BeamAttributes& ba, ZLIB::VTps& p)
{
  return TeapotEngine<ZLIB::Tps, ZLIB::VTps>::propagate(te, ba, p);
}

int TeapotDAIntegrator::propagate(const TeapotElement& te, ZLIB::VTps& p, ZLIB::VTps& tmp, 
				  PAC::BeamAttributes& ba, double* v0byc)
{
  return TeapotEngine<ZLIB::Tps, ZLIB::VTps>::propagate(te, p, tmp, ba, v0byc);
}

void TeapotDAIntegrator::passRfKick( int, ZLIB::VTps& , ZLIB::VTps& , PAC::BeamAttributes&, double*)
{
  // cerr << "TeapotDAIntegrator: this version does not support a RF kick" << endl;
}

int  TeapotDAIntegrator::testAperture(ZLIB::VTps& ) 
{ return 0; }
