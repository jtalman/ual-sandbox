// Library       : UAL
// File          : UAL/APF/PropagatorSequence.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include <typeinfo>
#include <iostream>
#include "UAL/APF/PropagatorSequence.hh"

UAL::AcceleratorNode UAL::PropagatorSequence::s_emptyAccNode;

UAL::PropagatorSequence::PropagatorSequence()
{
}

UAL::PropagatorSequence::PropagatorSequence(const UAL::PropagatorSequence& rhs)
{
  copy(rhs);
}

UAL::PropagatorSequence::~PropagatorSequence()
{
}

UAL::PropagatorSequence& UAL::PropagatorSequence::operator=(const UAL::PropagatorSequence& rhs)
{
  copy(rhs);
  return *this;
}

const char* UAL::PropagatorSequence::getType()
{
  return "UAL::PropagatorSequence";
}

bool UAL::PropagatorSequence::isSequence()
{
  return true;
}

UAL::AcceleratorNode& UAL::PropagatorSequence::getFrontAcceleratorNode()
{
  return s_emptyAccNode;
  // if(m_nodes.empty()) return s_emptyAccNode;
  // return m_nodes.front()->getFrontAcceleratorNode();
}

UAL::AcceleratorNode& UAL::PropagatorSequence::getBackAcceleratorNode()
{
  return s_emptyAccNode;
  // if(m_nodes.empty()) return s_emptyAccNode;
  // return m_nodes.back()->getBackAcceleratorNode();
}

void UAL::PropagatorSequence::setLatticeElements(const UAL::AcceleratorNode&, int, int, 
						 const UAL::AttributeSet&)
{
}


void UAL::PropagatorSequence::propagate(UAL::Probe& bunch)
{
  if(m_nodes.empty()) return;

  std::list<UAL::PropagatorNodePtr>::iterator i;
  // int counter = 0;
  for(i = m_nodes.begin(); i != m_nodes.end(); i++){
    // std::cout << "PropagatorSequence node id = " << counter++ << std::endl;
    (*i)->propagate(bunch);
  }

}

UAL::PropagatorIterator UAL::PropagatorSequence::begin()
{
  return m_nodes.begin();
}

UAL::PropagatorIterator UAL::PropagatorSequence::end()
{
  return m_nodes.end();
}

UAL::PropagatorNode* UAL::PropagatorSequence::clone()
{
  return new UAL::PropagatorSequence(*this);
}

int UAL::PropagatorSequence::size() const
{
  return m_nodes.size();
}

void UAL::PropagatorSequence::add(PropagatorNodePtr& nodePtr)
{
  m_nodes.push_back(nodePtr);
}

void UAL::PropagatorSequence::copy(const UAL::PropagatorSequence& rhs)
{
  m_nodes = rhs.m_nodes;
}
