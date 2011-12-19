//# Library       : SIMBAD
//# File          : SIMBAD/Base/BasicPropagator.cc
//# Copyright     : see Copyright file
//# Authors       : N.Malitsky & N.D'Imperio

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "SIMBAD/Base/BasicPropagator.hh"

const char*  SIMBAD::BasicPropagator::getType(){
  return "SIMBAD::BasicPropagator";
}


void SIMBAD::BasicPropagator::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						 int i0, int i1, 
						 const UAL::AttributeSet& attSet)
{

  if(i0 < sequence.getNodeCount()) m_frontNode = sequence.getNodeAt(i0);
  if(i1 < sequence.getNodeCount()) m_backNode  = sequence.getNodeAt(i1);
 
}

UAL::AcceleratorNode& SIMBAD::BasicPropagator::getFrontAcceleratorNode()
{
   if(m_frontNode == 0) return UAL::AcceleratorNodeFinder::getEmptyNode();
  else return *m_frontNode;
}

UAL::AcceleratorNode& SIMBAD::BasicPropagator::getBackAcceleratorNode()
{
  if(m_backNode == 0) return UAL::AcceleratorNodeFinder::getEmptyNode();
  else return *m_backNode;
}


