// Library       : UAL
// File          : UAL/SMF/AcceleratorNodeFinder.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include "UAL/SMF/AcceleratorNodeFinder.hh"

UAL::AcceleratorNodeFinder* UAL::AcceleratorNodeFinder::s_theInstance = 0;
UAL::AcceleratorNode UAL::AcceleratorNodeFinder::s_emptyAccNode;


UAL::AcceleratorNodeFinder::AcceleratorNodeFinder()
{
}

UAL::AcceleratorNodeFinder& UAL::AcceleratorNodeFinder::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new UAL::AcceleratorNodeFinder();
  }
  return *s_theInstance;
}

UAL::AcceleratorNode& UAL::AcceleratorNodeFinder::getEmptyNode()
{
  return s_emptyAccNode;
}


void UAL::AcceleratorNodeFinder::add(const UAL::AcceleratorNodePtr& nodePtr)
{
  UAL::AcceleratorNodeFinder::Iterator it = m_extent.find(nodePtr->getName());
  if(it == m_extent.end()){
    m_extent[nodePtr->getName()] = nodePtr;
  }  
}

void UAL::AcceleratorNodeFinder::clean()
{
  return m_extent.clear();
}

UAL::AcceleratorNodeFinder::Iterator UAL::AcceleratorNodeFinder::begin()
{
  return m_extent.begin();
}

UAL::AcceleratorNodeFinder::Iterator UAL::AcceleratorNodeFinder::find(const std::string& id)
{
  return m_extent.find(id);
}

UAL::AcceleratorNodeFinder::Iterator UAL::AcceleratorNodeFinder::end()
{
  return m_extent.end();
}

