// Library       : TEAPOT
// File          : TEAPOT/Integrator/BasicPropagator.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include "TEAPOT/Integrator/BasicPropagator.hh"

double TEAPOT::BasicPropagator::s_steps[] = {0.1, 4./15., 4./15., 4./15., 0.1};  

TEAPOT::BasicPropagator::BasicPropagator()
{
}

TEAPOT::BasicPropagator::~BasicPropagator()
{
}

UAL::PropagatorNode*  TEAPOT::BasicPropagator::clone()
{
  return new TEAPOT::BasicPropagator();
}

const char* TEAPOT::BasicPropagator::getType()
{
  return "TEAPOT::BasicPropagator";
}

void TEAPOT::BasicPropagator::propagate(UAL::Probe& probe)
{
}

void TEAPOT::BasicPropagator::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						 int i0, int i1, 
						 const UAL::AttributeSet& attSet)
{
  if(i0 < sequence.getNodeCount()) m_frontNode = 
				     *((PacLattElement*) sequence.getNodeAt(i0));
  if(i1 < sequence.getNodeCount()) m_backNode  = 
				     *((PacLattElement*) sequence.getNodeAt(i1));

}

UAL::AcceleratorNode& TEAPOT::BasicPropagator::getFrontAcceleratorNode()
{
  return m_frontNode;
}

UAL::AcceleratorNode& TEAPOT::BasicPropagator::getBackAcceleratorNode()
{
  return m_backNode;
}

