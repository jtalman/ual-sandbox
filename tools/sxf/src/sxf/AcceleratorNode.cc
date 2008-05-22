#include "sxf/AcceleratorNode.hh"

// Constructor.
SXF::AcceleratorNode::AcceleratorNode(SXF::OStream& out, const char* type)
  : m_refOStream(out)
{
  setType(type);
}

// Destructor
SXF::AcceleratorNode::~AcceleratorNode()
{
  if(m_sType) { delete [] m_sType; }
}

// Complete all operations.
void SXF::AcceleratorNode::update()
{
}

// Return this node reader to its initial conditions.
void SXF::AcceleratorNode::close()
{
}

// Set node type.
void SXF::AcceleratorNode::setType(const char* type) 
{
  m_sType = new char[strlen((char*) type) + 1];
  strcpy(m_sType, (char*)  type);
}
