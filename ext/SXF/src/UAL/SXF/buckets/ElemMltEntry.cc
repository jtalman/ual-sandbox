#include "UAL/SXF/buckets/ElemMltEntry.hh"

// Constructor.
UAL::SXFElemMltEntry::SXFElemMltEntry(SXF::OStream& out)
  : UAL::SXFElemMultipole(out, "entry")
{
}

// Open bucket: Set the deviation flag to true 
// if the bucketType is "entry.dev".
int UAL::SXFElemMltEntry::openObject(const char* type)
{
  if(!strcmp(type, "entry.dev")) m_iIsDeviation = 1;
  else m_iIsDeviation = 0;
  return SXF_TRUE;
}

// Return a number of SMF entry buckets.
int UAL::SXFElemMltEntry::getEntrySize() const
{
  return 1;
}

// Get the SMF body bucket selected by the given index (0).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL::SXFElemMltEntry::getEntryBucket(int index) 
{
  if(index) return 0;
  if(m_iOrderKL < 0 && m_iOrderKTL < 0) return 0;
  return m_pMultipole;
}

// Return entry attributes
// PacElemAttributes* UAL::SXFElemMltEntry::getAttributes(const PacLattElement& element)
// {
//  return element.getFront();
// }


// Return exit design multipole
PacElemMultipole UAL::SXFElemMltEntry::getDesignMultipole(const PacLattElement& element)
{
  PacElemMultipole result;

  PacElemPart* part = element.genElement().getFront();
  if(!part) return result;

  PacElemAttributes& attributes = part->attributes();

  PacElemAttributes::iterator it;

  // Check multipole bucket
  it = attributes.find(pacMultipole.key());
  if(!(it != attributes.end())) return result;

  return static_cast<PacElemMultipole>( (*it));  
}

// Return exit design multipole
PacElemMultipole UAL::SXFElemMltEntry::getTotalMultipole(const PacLattElement& element)
{
  PacElemMultipole result;

  PacElemAttributes* attributes = element.getFront();
  if(!attributes) return result;

  PacElemAttributes::iterator it;

  // Check multipole bucket
  it = attributes->find(pacMultipole.key());
  if(!(it != attributes->end())) return result;

  return (*it);  
}




