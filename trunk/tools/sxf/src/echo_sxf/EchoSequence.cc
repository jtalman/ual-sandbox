#include "echo_sxf/EchoSequence.hh"

// Constructor.
SXF::EchoSequence::EchoSequence(SXF::OStream& out)
  : SXF::Sequence(out) 
{
}

// Create and return a sequence echo writer.
SXF::Sequence* SXF::EchoSequence::clone()
{
  SXF::Sequence* seq = new SXF::EchoSequence(m_refOStream);
  return seq;
}

// Do nothing.
int SXF::EchoSequence::openObject(const char*, const char*)
{
  return SXF_TRUE;
}

// Do nothing.
void SXF::EchoSequence::update()
{
}

// Do nothing.
void SXF::EchoSequence::close()
{
}

// Do nothing. 
void SXF::EchoSequence::setDesign(const char*)
{
}

// Do nothing.
void SXF::EchoSequence::setLength(double)
{
}

// Do nothing.
void SXF::EchoSequence::setAt(double)
{
}

// Do nothing.
void SXF::EchoSequence::setHAngle(double)
{
}

// Do nothing.
void SXF::EchoSequence::addNode(SXF::AcceleratorNode* )
{
    
}
