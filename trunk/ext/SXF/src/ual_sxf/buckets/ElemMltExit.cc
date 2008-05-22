#include "ual_sxf/buckets/ElemMltExit.hh"

// Constructor.
UAL_SXF_ElemMltExit::UAL_SXF_ElemMltExit(SXF::OStream& out)
  : UAL_SXF_ElemMultipole(out, "exit")
{
}

// Return a number of SMF exit buckets.
int UAL_SXF_ElemMltExit::getExitSize() const
{
  return 1;
}

// Get the SMF exit bucket selected by the given index (0).
// Return 0, if the bucket is not defined or empty.
PacElemBucket* UAL_SXF_ElemMltExit::getExitBucket(int index) 
{
  if(index) return 0;
  if(m_iOrderKL < 0 && m_iOrderKTL < 0) return 0;
  return m_pMultipole;
}

// Return exit attributes
PacElemAttributes* UAL_SXF_ElemMltExit::getAttributes(const PacLattElement& element)
{
  return element.getEnd();
}

