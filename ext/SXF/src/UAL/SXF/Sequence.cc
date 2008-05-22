#include "UAL/SXF/Element.hh"
#include "UAL/SXF/Sequence.hh"

double atError = 3.e-4; // 1.e-8;

// Constructor.
UAL::SXFSequence::SXFSequence(SXF::OStream& out)
  : SXF::Sequence(out), 
    m_pLattice(0), 
    m_iDriftCounter(0),
    m_dPosition(0.0)
{
}

// Destructor.
UAL::SXFSequence::~SXFSequence()
{
  if(m_pLattice) { delete m_pLattice;}
}

// Create a sequence adaptor.
SXF::Sequence* UAL::SXFSequence::clone()
{
  SXF::Sequence* seq = new UAL::SXFSequence(m_refOStream);
  return seq;
}

// Create a SMF lattice.
int UAL::SXFSequence::openObject(const char* name, const char*)
{
  m_pLattice = new PacLattice(name);
  return SXF_TRUE;
}

// Initialize the lattice by the list of elements
void UAL::SXFSequence::update()
{
  if(!m_pLattice) return;
  m_pLattice->set(m_ElementList);
}

// Release data
void UAL::SXFSequence::close()
{

  // release local data

  m_iDriftCounter = 0;
  m_dPosition     = 0.0;

  if(m_pLattice) { delete m_pLattice;   m_pLattice = 0;}
  m_ElementList.erase(m_ElementList.begin(), m_ElementList.end());
  
}

// Do nothing.
void UAL::SXFSequence::setDesign(const char*)
{
}

// Do nothing.
void UAL::SXFSequence::setLength(double)
{
}

// Do nothing.
void UAL::SXFSequence::setAt(double)
{
}

// Do nothing.
void UAL::SXFSequence::setHAngle(double)
{
}

// Add node
void UAL::SXFSequence::addNode(SXF::AcceleratorNode* node)
{

  if(!node) return;
 

  // Check if it is a sequence 

  if(!node->isElement()){ 
    cerr << "Error: this version does not support nested sequences " 
	 << endl;
  }

  // Prepare the SXF element adaptor

  UAL::SXFElement *eparser = (UAL::SXFElement*) node;

  // Define the SMF element.

  double nodeAt = eparser->getAt();
  double nodeL  = eparser->getLength();
  PacLattElement* element = eparser->getLattElement();

  if(!element) return;

  // Define a drift between previous and present elements if
  // the present element position > the current position, and
  // name it "<lattice name>_<drift counter>".

  double driftL = 0;
  double at = 0.0;
  PacLattElement drift;  

  at = nodeAt;
  driftL = nodeAt - m_dPosition - nodeL/2.; 

  /*
  if(nodeAt == 0){ 
    at = m_dPosition + nodeL/2.;
  }
  else{
    if(nodeAt >= m_dPosition ) {
      at = nodeAt;
      driftL = nodeAt - m_dPosition - nodeL/2.;

      // m_DriftLength.l(driftL);
      // char sCounter[5];
      // sprintf(sCounter, "%d", m_iDriftCounter++);
      // drift.name("_" + m_pLattice->name() + "_" + sCounter);
      // drift.set(m_DriftLength);
    }
    // else if ((nodeAt + atError) >= m_dPosition ) {
    //  at = nodeAt;
    // }
    else {
      m_refOStream.cfe_error() 
	<< "\n *** UAL/SXF Error : Sequence::addNode : node(" 
	<< element->name() << ") at - (position + l/2) = " 
	<< nodeAt - m_dPosition << " < 0" 
	<< endl;
    }
  }
  */

  // Update the list of elements

  // if(driftL > 0){ m_ElementList.push_back(drift); }

  if(driftL > atError){  
      m_DriftLength.l(driftL);
      char sCounter[5];
      sprintf(sCounter, "%d", m_iDriftCounter++);
      drift.name("_" + m_pLattice->name() + "_" + sCounter);
      // std::cout << "SXF adds drift at = " << at << "driftL = " << driftL << " counter = " << sCounter << std::endl;
      drift.set(m_DriftLength);
      m_ElementList.push_back(drift); 
  }
  else if (driftL < -atError) {
      m_refOStream.cfe_error() 
	<< "\n *** UAL/SXF Error : Sequence::addNode : node(" 
	<< element->name() << ") at - (position + l/2) = " 
	<< nodeAt - m_dPosition << " < 0" 
	<< endl;
  }
  m_ElementList.push_back(*element);  

  // Calculate the current position 
  // m_dPosition += driftL + nodeL;
  m_dPosition = at + nodeL/2.;
    
}


