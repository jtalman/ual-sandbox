// Library       : TEAPOT
// File          : TEAPOT/Integrator/MltDaIntegrator.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "Optics/PacTMap.h"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "TEAPOT/Integrator/MltDaIntegrator.hh"

TEAPOT::MagnetAlgorithm<ZLIB::Tps, ZLIB::VTps> TEAPOT::MltDaIntegrator::s_algorithm;

const char* TEAPOT::MltDaIntegrator::getType()
{
  return "TEAPOT::MltDaIntegrator";
}

TEAPOT::MltDaIntegrator::MltDaIntegrator()
{
  initialize();
}

TEAPOT::MltDaIntegrator::MltDaIntegrator(const TEAPOT::MltDaIntegrator& di)
{
  copy(di);
}

TEAPOT::MltDaIntegrator::~MltDaIntegrator()
{
}

UAL::PropagatorNode* TEAPOT::MltDaIntegrator::clone()
{
  return new TEAPOT::MltDaIntegrator(*this);
}

void TEAPOT::MltDaIntegrator::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						 int is0, 
						 int is1,
						 const UAL::AttributeSet& attSet)
{
   TEAPOT::BasicPropagator::setLatticeElements(sequence, is0, is1, attSet);
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}


void TEAPOT::MltDaIntegrator::setLatticeElement(const PacLattElement& e)
{
  // length
  m_l = e.getLength();

  // ir
  m_ir = e.getN();

  m_mdata.setLatticeElement(e);
}

void TEAPOT::MltDaIntegrator::propagate(UAL::Probe& probe)
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

  // Simple Element

  if(!m_ir){
    s_algorithm.passDrift(m_l/2., p, tmp, v0byc);
    s_algorithm.applyMltKick(m_mdata, 1., p); 
    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.passDrift(m_l/2., p, tmp, v0byc);
    s_algorithm.passExit(m_mdata, p); 
    return;
  } 

  // Complex Element


  double rIr = 1./m_ir;
  double rkicks = 0.25*rIr;

  int counter = 0;
  for(int i = 0; i < m_ir; i++){
    for(int is = 0; is < 4; is++){
      counter++;
      s_algorithm.passDrift(m_l*s_steps[is]*rIr, p, tmp, v0byc);
      s_algorithm.applyMltKick(m_mdata, rkicks, p); 
      s_algorithm.makeVelocity(p, tmp, v0byc);	
    }
    counter++;
    s_algorithm.passDrift(m_l*s_steps[4]*rIr, p, tmp, v0byc); 
 }

 s_algorithm.passExit(m_mdata, p); 
 // testAperture(p);

}

void TEAPOT::MltDaIntegrator::initialize()
{
  m_l = 0.0;
  m_ir = 0.0;
}

void TEAPOT::MltDaIntegrator::copy(const TEAPOT::MltDaIntegrator& mt)
{
  m_l   = mt.m_l;
  m_ir  = mt.m_ir;

  m_mdata = mt.m_mdata;
}








