#include <stdlib.h>
// #include <iostream.h>
#include "sxf/AcceleratorReader.hh"

SXF::AcceleratorReader* SXF::AcceleratorReader::s_reader = 0;

// Constructor.
SXF::AcceleratorReader::AcceleratorReader(SXF::OStream& out, int stackSize) 
  : 
  m_refOStream(out), 
  m_pNodeRegistry(0), 
  m_pSequence(0), 
  m_SequenceStack(stackSize),  
  m_pElement(0) 
{}

// Destructor.
SXF::AcceleratorReader::~AcceleratorReader()
{
}

// Prepare this accelerator reader for operations.
// Push a root sequence reader to the stack and make it current.
int SXF::AcceleratorReader::open(const char* name)
{
  m_pSequence = m_pNodeRegistry->getSequence()->clone();
  if(!m_pSequence) {
    m_refOStream.cfe_error() 
      << "\n*** CFE Error: SXF::AcceleratorReader::open(char* ) : " 
      << "root sequence has not be created " << endl;
    exit(1);
  }
  m_pSequence->openObject(name, "sequence");
  m_SequenceStack.push(m_pSequence);
  return SXF_TRUE;
}

// Return an accelerator reader to its initial conditions.
void SXF::AcceleratorReader::close()
{  
  m_SequenceStack.pop();
  m_pSequence->update();
  m_pSequence->close();
  delete m_pSequence;
  m_pSequence = 0;
}

// Push a sequence reader to the stack and make it current.
int SXF::AcceleratorReader::openSequence(const char* name)
{
  m_pSequence = m_pNodeRegistry->getSequence()->clone();
  if(!m_pSequence) {
    m_refOStream.cfe_error() 
      << "\n*** CFE Error: SXF::AcceleratorReader::openSequence(char* ) : " 
      << "sequence " << name << " has not be created " << endl;
    exit(1);
  }  
  m_pSequence->openObject(name, "sequence");
  m_SequenceStack.push(m_pSequence);
  return SXF_TRUE;
}

// Return a current sequence reader.
SXF::Sequence* SXF::AcceleratorReader::getSequence()
{
  return m_pSequence;
}

// Close a current sequence reader and pop it from the stack.
void SXF::AcceleratorReader::closeSequence()
{
  m_SequenceStack.pop();
  m_pSequence->update();
  m_SequenceStack.top()->addNode(m_pSequence);
  m_pSequence->close();
  delete m_pSequence;
  m_pSequence = m_SequenceStack.top();
}

// Select an element reader and make it current.
int SXF::AcceleratorReader::openElement(const char* type, const char* name)
{
  m_pElement = m_pNodeRegistry->getElement(type);
  if(!m_pElement) {  
    m_pElement = m_pNodeRegistry->getErrorElement(); 
    m_pElement->openObject(name, type);
    return SXF_FALSE;
  }
  return m_pElement->openObject(name, type);
}

// Return a current element reader.
SXF::Element* SXF::AcceleratorReader::getElement()
{
  return m_pElement;
}

// Close a current element reader.
void SXF::AcceleratorReader::closeElement()
{
  m_pElement->update();
  m_pSequence->addNode(m_pElement);
  m_pElement->close();
  m_pElement = 0;
}

// Return Ostream.
SXF::OStream& SXF::AcceleratorReader::echo()
{
  return m_refOStream;
}

