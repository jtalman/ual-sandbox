#include "UAL/SXF/buckets/ElemMltExit.hh"

// Constructor.
UAL::SXFElemMltExit::SXFElemMltExit(SXF::OStream& out)
  : UAL::SXFElemMultipole(out, "exit")
{
}

// Open bucket: Set the deviation flag to true 
// if the bucketType is "exit.dev".
int UAL::SXFElemMltExit::openObject(const char* type)
{
  if(!strcmp(type, "exit.dev")) m_iIsDeviation = 1;
  else m_iIsDeviation = 0;
  return SXF_TRUE;
}

// Return a number of SMF exit buckets.
int UAL::SXFElemMltExit::getExitSize() const
{
  return 1;
}

// Get the SMF exit bucket selected by the given index (0).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL::SXFElemMltExit::getExitBucket(int index) 
{
  if(index) return 0;
  if(m_iOrderKL < 0 && m_iOrderKTL < 0) return 0;
  return m_pMultipole;
}

// Return exit attributes
// PacElemAttributes* UAL::SXFElemMltExit::getAttributes(const PacLattElement& element)
// {
//  return element.getEnd();
// }

// Return exit design multipole
PacElemMultipole UAL::SXFElemMltExit::getDesignMultipole(const PacLattElement& element)
{
  PacElemMultipole result;

  PacElemPart* part = element.genElement().getEnd();
  if(!part) return result;

  PacElemAttributes& attributes = part->attributes();

  PacElemAttributes::iterator it;

  // Check multipole bucket
  it = attributes.find(pacMultipole.key());
  if(!(it != attributes.end())) return result;

  return static_cast<PacElemMultipole>( (*it));  
}

// Return exit design multipole
PacElemMultipole UAL::SXFElemMltExit::getTotalMultipole(const PacLattElement& element)
{
  PacElemMultipole result;

  PacElemAttributes* attributes = element.getEnd();
  if(!attributes) return result;

  PacElemAttributes::iterator it;

  // Check multipole bucket
  it = attributes->find(pacMultipole.key());
  if(!(it != attributes->end())) return result;

  return (*it);  
}


