// Library       : TIBETAN
// File          : TIBETAN/Propagator/BasicPropagator.cc
// Copyright     : see Copyright file
// Author        : J.Wei
// C++ version   : N.Malitsky 

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "TIBETAN/Propagator/BasicPropagator.hh"

const char* TIBETAN::BasicPropagator::getType()
{
  return "TIBETAN::BasicPropagator";
}

TIBETAN::BasicPropagator::BasicPropagator()
{
}

TIBETAN::BasicPropagator::~BasicPropagator()
{
}

void TIBETAN::BasicPropagator::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						  int i0, int i1, 
						  const UAL::AttributeSet& attSet)
{

  m_accName = sequence.getName();
  m_i0 = i0;
  m_i1 = i1;  

  // if(m_i0 < sequence.getNodeCount()) m_frontNode = *((PacLattElement*) sequence.getNodeAt(i0));
  // if(m_i1 < sequence.getNodeCount()) m_backNode  = *((PacLattElement*) sequence.getNodeAt(i1));
  if(i0 < sequence.getNodeCount()) m_frontNode = sequence.getNodeAt(i0);
  if(i1 < sequence.getNodeCount()) m_backNode  = sequence.getNodeAt(i1);

}

UAL::AcceleratorNode& TIBETAN::BasicPropagator::getFrontAcceleratorNode()
{
  /*
  UAL::AcceleratorNodeFinder::Iterator it = 
    UAL::AcceleratorNodeFinder::getInstance().find(m_accName);

  if(it == UAL::AcceleratorNodeFinder::getInstance().end()){
    return s_emptyAccNode;
  } 
  
  int size = it->second->getNodeCount();
  if(m_i0 >= size) return s_emptyAccNode;

  return *(it->second->getNodeAt(m_i0));
  */
  // return m_frontNode;
  if(m_frontNode == 0) return UAL::AcceleratorNodeFinder::getEmptyNode();
  else return *m_frontNode;
}

UAL::AcceleratorNode& TIBETAN::BasicPropagator::getBackAcceleratorNode()
{
  /*
  UAL::AcceleratorNodeFinder::Iterator it = 
    UAL::AcceleratorNodeFinder::getInstance().find(m_accName);

  if(it == UAL::AcceleratorNodeFinder::getInstance().end()){
    return s_emptyAccNode;
  } 
  
  int size = it->second->getNodeCount();
  if(m_i0 >= size) return s_emptyAccNode;

  return *(it->second->getNodeAt(m_i1));
  */
  // return m_backNode;
  if(m_backNode == 0) return UAL::AcceleratorNodeFinder::getEmptyNode();
  else return *m_backNode;
}

void TIBETAN::BasicPropagator::propagate(UAL::Probe& probe)
{
}
