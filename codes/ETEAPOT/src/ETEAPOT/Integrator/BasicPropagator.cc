// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/BasicPropagator.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include "EETEAPOT/Integrator/BasicPropagator.hh"

double ETEAPOT::BasicPropagator::s_steps[] = {0.1, 4./15., 4./15., 4./15., 0.1};  

ETEAPOT::BasicPropagator::BasicPropagator()
{
}

ETEAPOT::BasicPropagator::~BasicPropagator()
{
}

UAL::PropagatorNode*  ETEAPOT::BasicPropagator::clone()
{
  return new ETEAPOT::BasicPropagator();
}

void ETEAPOT::BasicPropagator::propagate(UAL::Probe& probe)
{
}

void ETEAPOT::BasicPropagator::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						 int i0, int i1, 
						 const UAL::AttributeSet& attSet)
{
  if(i0 < sequence.getNodeCount()) m_frontNode = 
				     *((PacLattElement*) sequence.getNodeAt(i0));
  if(i1 < sequence.getNodeCount()) m_backNode  = 
				     *((PacLattElement*) sequence.getNodeAt(i1));

}

UAL::AcceleratorNode& ETEAPOT::BasicPropagator::getFrontAcceleratorNode()
{
  return m_frontNode;
}

UAL::AcceleratorNode& ETEAPOT::BasicPropagator::getBackAcceleratorNode()
{
  return m_backNode;
}

