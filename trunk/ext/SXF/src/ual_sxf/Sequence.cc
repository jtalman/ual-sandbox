#include "ual_sxf/Element.hh"
#include "ual_sxf/Sequence.hh"

double atError = 1.e-8;

// Constructor.
UAL_SXF_Sequence::UAL_SXF_Sequence(SXF::OStream& out)
  : SXF::Sequence(out), 
    m_pLattice(0), 
    m_iDriftCounter(0),
    m_dPosition(0.0)
{
}

// Destructor.
UAL_SXF_Sequence::~UAL_SXF_Sequence()
{
  if(m_pLattice) { delete m_pLattice;}
}

// Create a sequence adaptor.
SXF::Sequence* UAL_SXF_Sequence::clone()
{
  SXF::Sequence* seq = new UAL_SXF_Sequence(m_refOStream);
  return seq;
}

// Create a SMF lattice.
int UAL_SXF_Sequence::openObject(const char* name, const char*)
{
  m_pLattice = new PacLattice(name);
  return SXF_TRUE;
}

// Initialize the lattice by the list of elements
void UAL_SXF_Sequence::update()
{
  if(!m_pLattice) return;
  m_pLattice->set(m_ElementList);
}

// Release data
void UAL_SXF_Sequence::close()
{

  // release local data

  m_iDriftCounter = 0;
  m_dPosition     = 0.0;
  if(m_pLattice) { delete m_pLattice;   m_pLattice = 0;}
  m_ElementList.erase(m_ElementList.begin(), m_ElementList.end());
  
}

// Do nothing.
void UAL_SXF_Sequence::setDesign(const char*)
{
}

// Do nothing.
void UAL_SXF_Sequence::setLength(double)
{
}

// Do nothing.
void UAL_SXF_Sequence::setAt(double)
{
}

// Do nothing.
void UAL_SXF_Sequence::setHAngle(double)
{
}

// Add node
void UAL_SXF_Sequence::addNode(SXF::AcceleratorNode* node)
{

  if(!node) return;
 

  // Check if it is a sequence 

  if(!node->isElement()){ 
    cerr << "Error: this version does not support nested sequences " 
	 << endl;
  }

  // Prepare the SXF element adaptor

  UAL_SXF_Element *eparser = (UAL_SXF_Element*) node;

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

  if(nodeAt == 0){ 
    at = m_dPosition + nodeL/2.;
  }
  else{
    if(nodeAt >= m_dPosition ) {
      at = nodeAt;
      driftL = nodeAt - m_dPosition - nodeL/2.;
      m_DriftLength.l(driftL);
      char sCounter[5];
      sprintf(sCounter, "%d", m_iDriftCounter++);
      drift.name("_" + m_pLattice->name() + "_" + sCounter);
      drift.set(m_DriftLength);
    }
    else if ((nodeAt + atError) >= m_dPosition ) {
      at = nodeAt;
    }
    else {
      m_refOStream.cfe_error() 
	<< "\n *** UAL/SXF Error : Sequence::addNode : node(" 
	<< element->name() << ") at - (position + l/2) = " 
	<< nodeAt - m_dPosition << " < 0" 
	<< endl;
    }
  }

  // Update the list of elements

  if(driftL > 0){ m_ElementList.push_back(drift); }
  m_ElementList.push_back(*element);  

  // Calculate the current position 
  // m_dPosition += driftL + nodeL;
  m_dPosition = at + nodeL/2.;
    
}


