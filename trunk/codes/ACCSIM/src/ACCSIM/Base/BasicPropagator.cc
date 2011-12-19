//# Library       : ACCSIM
//# File          : ACCSIM/Base/BasicPropagator.cc
//# Copyright     : see Copyright file
//# Author        : F.W.Jones
//# C++ version   : N.Malitsky 

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "ACCSIM/Base/BasicPropagator.hh"

const char* ACCSIM::BasicPropagator::getType()
{
  return "ACCSIM::BasicPropagator";
}

ACCSIM::BasicPropagator::BasicPropagator()
{
}

ACCSIM::BasicPropagator::~BasicPropagator()
{
}

void ACCSIM::BasicPropagator::propagate(UAL::Probe&)
{
}

UAL::PropagatorNode*  ACCSIM::BasicPropagator::clone()
{
  return new ACCSIM::BasicPropagator();
}

void ACCSIM::BasicPropagator::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						 int i0, int i1, 
						 const UAL::AttributeSet& attSet)
{

  if(i0 < sequence.getNodeCount()) m_frontNode = sequence.getNodeAt(i0);
  if(i1 < sequence.getNodeCount()) m_backNode  = sequence.getNodeAt(i1);
 
}

UAL::AcceleratorNode& ACCSIM::BasicPropagator::getFrontAcceleratorNode()
{
   if(m_frontNode == 0) return UAL::AcceleratorNodeFinder::getEmptyNode();
  else return *m_frontNode;
}

UAL::AcceleratorNode& ACCSIM::BasicPropagator::getBackAcceleratorNode()
{
  if(m_backNode == 0) return UAL::AcceleratorNodeFinder::getEmptyNode();
  else return *m_backNode;
}

