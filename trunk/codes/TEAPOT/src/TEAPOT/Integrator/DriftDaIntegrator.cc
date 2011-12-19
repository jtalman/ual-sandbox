// Library       : TEAPOT
// File          : TEAPOT/Integrator/DriftDaIntegrator.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "Optics/PacTMap.h"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "TEAPOT/Integrator/DriftDaIntegrator.hh"

TEAPOT::DriftAlgorithm<ZLIB::Tps, ZLIB::VTps> TEAPOT::DriftDaIntegrator::s_algorithm;

const char* TEAPOT::DriftDaIntegrator::getType()
{
  return "TEAPOT::DriftDaIntegrator";
}

TEAPOT::DriftDaIntegrator::DriftDaIntegrator()
{
  initialize();
}

TEAPOT::DriftDaIntegrator::DriftDaIntegrator(const TEAPOT::DriftDaIntegrator& dt)
{
  copy(dt);
}

TEAPOT::DriftDaIntegrator::~DriftDaIntegrator()
{
}

UAL::PropagatorNode* TEAPOT::DriftDaIntegrator::clone()
{
  return new TEAPOT::DriftDaIntegrator(*this);
}

void TEAPOT::DriftDaIntegrator::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					      int is0, 
					      int is1,
					      const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicPropagator::setLatticeElements(sequence, is0, is1, attSet);  
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void TEAPOT::DriftDaIntegrator::setLatticeElement(const PacLattElement& e)
{
  m_l = e.getLength();
}

void TEAPOT::DriftDaIntegrator::propagate(UAL::Probe& probe)
{
  PacTMap& map = static_cast<PacTMap&>(probe);
  
  PAC::BeamAttributes& ba = map.getBeamAttributes();

  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);

  double v0byc = p0/e0;

  ZLIB::VTps tmp;

  ZLIB::VTps& p = map.operator*();
  tmp = p;
  s_algorithm.makeVelocity(p, tmp, v0byc);
  s_algorithm.makeRV(p, tmp, e0, p0, m0);
  s_algorithm.passDrift(m_l, p, tmp, v0byc);

}

void TEAPOT::DriftDaIntegrator::initialize()
{
  m_l = 0.0;
}

void TEAPOT::DriftDaIntegrator::copy(const TEAPOT::DriftDaIntegrator& dt)
{
  m_l = dt.m_l;
}

