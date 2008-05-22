// Library       : TEAPOT
// File          : TEAPOT/Integrator/MapDaIntegrator.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "TEAPOT/Integrator/MapDaIntegrator.hh"

PacLattice TEAPOT::MapDaIntegrator::s_lattice;
Teapot     TEAPOT::MapDaIntegrator::s_teapot;

TEAPOT::MapDaIntegrator::MapDaIntegrator()
{
  m_l = 0.0;
  m_map = 0;
}

TEAPOT::MapDaIntegrator::MapDaIntegrator(const TEAPOT::MapDaIntegrator& mi)
{
  m_l = mi.m_l;

  if(mi.m_map == 0) m_map = 0;
  else setMap(*mi.m_map);

}

TEAPOT::MapDaIntegrator::~MapDaIntegrator()
{
 if(m_map != 0) delete m_map;
}

UAL::PropagatorNode* TEAPOT::MapDaIntegrator::clone()
{
  return new TEAPOT::MapDaIntegrator(*this);
}

void TEAPOT::MapDaIntegrator::setLatticeElements(const UAL::AcceleratorNode& sequence, int is0, int is1,
						const UAL::AttributeSet& attSet)
{
  TEAPOT::BasicPropagator::setLatticeElements(sequence, is0, is1, attSet);
  const PacLattice& lattice     = (PacLattice&) sequence;
  PAC::BeamAttributes ba = (PAC::BeamAttributes&) attSet;

  // Initialize lattice 
  if(s_lattice.name() != lattice.name()){
    s_lattice = lattice;
    s_teapot.use(lattice);
  }

  // Calculate length
  m_l = 0;
  for(int i = is0; i < is1; i++){
    m_l += lattice[i].getLength(); 
  }

  // Propagate the sector map
  PacTMap sectorMap(6);
  s_teapot.trackMap(sectorMap, ba, is0, is1); 

  setMap(sectorMap);
}


void TEAPOT::MapDaIntegrator::setMap(const PacVTps& vtps)
{
  if(m_map == 0) m_map = new PacTMap(6);
  *m_map = vtps;
}

void TEAPOT::MapDaIntegrator::propagate(UAL::Probe& probe)
{
  PacTMap& map = static_cast<PacTMap&>(probe);  
  if(m_map != 0) m_map->propagate(map);
}

