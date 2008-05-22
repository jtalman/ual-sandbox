#include "ual_sxf/buckets/ElemMltEntry.hh"

// Constructor.
UAL_SXF_ElemMltEntry::UAL_SXF_ElemMltEntry(SXF::OStream& out)
  : UAL_SXF_ElemMultipole(out, "entry")
{
}

// Return a number of SMF entry buckets.
int UAL_SXF_ElemMltEntry::getEntrySize() const
{
  return 1;
}

// Get the SMF body bucket selected by the given index (0).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL_SXF_ElemMltEntry::getEntryBucket(int index) 
{
  if(index) return 0;
  if(m_iOrderKL < 0 && m_iOrderKTL < 0) return 0;
  return m_pMultipole;
}

// Return entry attributes
PacElemAttributes* UAL_SXF_ElemMltEntry::getAttributes(const PacLattElement& element)
{
  return element.getFront();
}

