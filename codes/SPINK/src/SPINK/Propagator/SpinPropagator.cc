// Library       : SPINK
// File          : SPINK/Propagator/SpinPropagator.cc
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#include "SPINK/Propagator/SpinPropagator.hh"

double SPINK::SpinPropagator::OTs_mat[3][3] = { 1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0 }; //AULNLD:11DEC09

const char* SPINK::SpinPropagator::getType()
{
  return "SPINK::SpinPropagator";
}

SPINK::SpinPropagator::SpinPropagator()
{

}

SPINK::SpinPropagator::~SpinPropagator()
{
}

void SPINK::SpinPropagator::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					       int i0, int i1, 
					       const UAL::AttributeSet& attSet)
{

  if(i0 < sequence.getNodeCount())
    m_frontNode = *((PacLattElement*) sequence.getNodeAt(i0));
  if(i1 < sequence.getNodeCount()) 
    m_backNode  = *((PacLattElement*) sequence.getNodeAt(i1));
}

UAL::AcceleratorNode& SPINK::SpinPropagator::getFrontAcceleratorNode()
{
  return m_frontNode;
}

UAL::AcceleratorNode& SPINK::SpinPropagator::getBackAcceleratorNode()
{
  return m_backNode;
}

void SPINK::SpinPropagator::propagate(UAL::Probe& b)
{
}
