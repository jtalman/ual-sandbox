#include "ual_sxf/ElemBucketRegistry.hh"
#include "ual_sxf/buckets/ElemError.hh"
#include "ual_sxf/buckets/ElemMltEntry.hh"
#include "ual_sxf/buckets/ElemMltExit.hh"
#include "ual_sxf/buckets/ElemAlign.hh"
#include "ual_sxf/buckets/ElemAperture.hh"

UAL_SXF_ElemBucketRegistry* UAL_SXF_ElemBucketRegistry::s_pBucketRegistry = 0;

// Constructor.
UAL_SXF_ElemBucketRegistry::UAL_SXF_ElemBucketRegistry(SXF::OStream& out) 
  : SXF::ElemBucketRegistry(out)
{
  allocateRegistry();

  m_pErrorBucket = new UAL_SXF_ElemError(out);

  bind("entry", new UAL_SXF_ElemMltEntry(out));
  bind("exit",  new UAL_SXF_ElemMltExit(out));
  bind("align", new UAL_SXF_ElemAlign(out));
  bind("aperture", new UAL_SXF_ElemAperture(out));  //make the aperture round by default
}

// Destructor.
UAL_SXF_ElemBucketRegistry::~UAL_SXF_ElemBucketRegistry() {
  if(m_aBuckets){
    for(int i=0; i < m_iSize; i++){ 
      if(m_aBuckets[i]) { delete m_aBuckets[i]; }
    }  
    delete [] m_aBuckets;
  }
  if(m_pErrorBucket) { delete m_pErrorBucket; }
}

// Return singleton.
UAL_SXF_ElemBucketRegistry* UAL_SXF_ElemBucketRegistry::getInstance(SXF::OStream& out)
{
  if(!s_pBucketRegistry) { 
    s_pBucketRegistry = new UAL_SXF_ElemBucketRegistry(out);
  }
  return s_pBucketRegistry;
}

// Write data.
void UAL_SXF_ElemBucketRegistry::write(ostream& out, const PacLattElement& element, const string& tab)
{
  UAL_SXF_ElemBucket* writer;
  for(int i=0; i < m_iSize; i++){ 
    writer = (UAL_SXF_ElemBucket*) m_aBuckets[i];
    if(writer) writer->write(out, element, tab);  
  }  
}
