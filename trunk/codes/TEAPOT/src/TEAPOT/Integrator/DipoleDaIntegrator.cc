// Library       : TEAPOT
// File          : TEAPOT/Integrator/DipoleDaIntegrator.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "Optics/PacTMap.h"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "TEAPOT/Integrator/DipoleDaIntegrator.hh"

TEAPOT::DipoleAlgorithm<ZLIB::Tps, ZLIB::VTps> TEAPOT::DipoleDaIntegrator::s_algorithm;

const char* TEAPOT::DipoleDaIntegrator::getType() {
  return "TEAPOT::DipoleAlgorithm";
}

TEAPOT::DipoleDaIntegrator::DipoleDaIntegrator()
{
}


TEAPOT::DipoleDaIntegrator::DipoleDaIntegrator(const TEAPOT::DipoleDaIntegrator& dt)
{
  m_data  = dt.m_data;
  m_mdata = dt.m_mdata;
}

TEAPOT::DipoleDaIntegrator::~DipoleDaIntegrator()
{
}

UAL::PropagatorNode* TEAPOT::DipoleDaIntegrator::clone()
{
  return new TEAPOT::DipoleDaIntegrator(*this);
}

void TEAPOT::DipoleDaIntegrator::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					       int is0, 
					       int is1,
					       const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicPropagator::setLatticeElements(sequence, is0, is1, attSet);
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}


void TEAPOT::DipoleDaIntegrator::setLatticeElement(const PacLattElement& e)
{
  m_data.setLatticeElement(e);
  m_mdata.setLatticeElement(e);
}

void TEAPOT::DipoleDaIntegrator::propagate(UAL::Probe& probe)
{
  PacTMap& map = static_cast<PacTMap&>(probe);
  
  PAC::BeamAttributes& ba = map.getBeamAttributes();

  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);

  double v0byc = p0/e0;

  ZLIB::VTps tmp;

  ZLIB::VTps& p = map.operator*();
  tmp = p;
  s_algorithm.passEntry(m_mdata, p);
  s_algorithm.makeVelocity(p, tmp, v0byc);
  s_algorithm.makeRV(p, tmp, e0, p0, m0);
  s_algorithm.passBend(m_data, m_mdata, p, tmp, v0byc);
  s_algorithm.passExit(m_mdata, p);
  // testAperture(p);

}










