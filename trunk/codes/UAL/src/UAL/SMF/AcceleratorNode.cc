// Library       : UAL
// File          : UAL/SMF/AcceleratorNode.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include "UAL/SMF/AcceleratorNode.hh"

std::string UAL::AcceleratorNode::s_emptyName("");

UAL::AcceleratorNode::AcceleratorNode()
{
}

UAL::AcceleratorNode::~AcceleratorNode()
{
}
/*
void UAL::AcceleratorNode::setAcceleratorComponent(const UAL::AcceleratorComponentPtr& component) 
{
  m_componentPtr = component;
}

const UAL::AcceleratorComponentPtr& UAL::AcceleratorNode::getAcceleratorComponent() const
{
  return m_componentPtr;
}
*/
const std::string& UAL::AcceleratorNode::getType() const
{
  return s_emptyName;
}

const std::string& UAL::AcceleratorNode::getName() const
{
  return s_emptyName;
}

const std::string& UAL::AcceleratorNode::getDesignName() const
{
  return s_emptyName;
}

double UAL::AcceleratorNode::getLength() const
{
  return 0.0;
}

double UAL::AcceleratorNode::getPosition() const
{
  return 0.0;
}

void UAL::AcceleratorNode::setPosition(double)
{
}

int UAL::AcceleratorNode::getNodeCount() const
{
  return 0;
}

UAL::AcceleratorNode* const UAL::AcceleratorNode::getNodeAt(int indx) const
{
  return 0;
}

UAL::AcceleratorNode* UAL::AcceleratorNode::clone() const
{
  return new UAL::AcceleratorNode();
}
