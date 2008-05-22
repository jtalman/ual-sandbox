// Library       : UAL
// File          : UAL/APF/AcceleratorPropagator.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include <pcre.h>
#include "UAL/APF/AcceleratorPropagator.hh"

UAL::AcceleratorPropagator::AcceleratorPropagator()
{
}

UAL::AcceleratorPropagator::~AcceleratorPropagator()
{
}

void UAL::AcceleratorPropagator::setName(const std::string& name)
{
  m_name = name;
}

const std::string& UAL::AcceleratorPropagator::getName() const
{
  return m_name;
}

UAL::PropagatorSequence& UAL::AcceleratorPropagator::getRootNode()
{
  return m_rootNode;
}

std::list<UAL::PropagatorNodePtr> 
UAL::AcceleratorPropagator::getNodesByName(const std::string& pattern) 
{
  std::list<UAL::PropagatorNodePtr> nodes;
  if (m_rootNode.size() == 0) return nodes;

  const char *error;
  int erroffset;
  pcre* re = pcre_compile(pattern.c_str(), 0, &error, &erroffset, NULL);
  pcre_extra* pe = pcre_study(re, 0, &error);

  int overtor[30]; // for regular expression

  UAL::PropagatorIterator it;
  for(it = m_rootNode.begin(); it != m_rootNode.end(); it++){
    UAL::AcceleratorNode& acNode = (*it)->getFrontAcceleratorNode();

    // Check the element name
    int rc = pcre_exec(re, pe, acNode.getName().c_str(), 
		       acNode.getName().size(), 0, 0, overtor, 30); 

    if(rc > 0) {  nodes.push_back(*it); }

  }

  return nodes;
  
}

void UAL::AcceleratorPropagator::propagate(UAL::Probe& probe)
{
  m_rootNode.propagate(probe);
}


