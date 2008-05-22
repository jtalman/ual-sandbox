// Library       : ICE
// File          : ICE/Base/BasicPropagator.cc
// Copyright     : see Copyright file
// Author        : M.Blaskiewicz
// C++ version   : N.Malitsky 

#include "ICE/Base/BasicPropagator.hh"


ICE::BasicPropagator::BasicPropagator()
{
}

ICE::BasicPropagator::~BasicPropagator()
{
}

void ICE::BasicPropagator::propagate(UAL::Probe& probe)
{
}

void ICE::BasicPropagator::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						 int i0, int i1, 
						 const UAL::AttributeSet& attSet)
{
  if(i0 < sequence.getNodeCount()) m_frontNode = *((PacLattElement*) sequence.getNodeAt(i0));
  if(i1 < sequence.getNodeCount()) m_backNode  = *((PacLattElement*) sequence.getNodeAt(i1));

}

UAL::AcceleratorNode& ICE::BasicPropagator::getFrontAcceleratorNode()
{
  return m_frontNode;
}

UAL::AcceleratorNode& ICE::BasicPropagator::getBackAcceleratorNode()
{
  return m_backNode;
}

