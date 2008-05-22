#include "UAL/SXF/ElemBucketRegistry.hh"
#include "UAL/SXF/buckets/ElemError.hh"
#include "UAL/SXF/buckets/ElemMltEntry.hh"
#include "UAL/SXF/buckets/ElemMltExit.hh"
#include "UAL/SXF/buckets/ElemAlign.hh"
#include "UAL/SXF/buckets/ElemAperture.hh"

UAL::SXFElemBucketRegistry* UAL::SXFElemBucketRegistry::s_pBucketRegistry = 0;

// Constructor.
UAL::SXFElemBucketRegistry::SXFElemBucketRegistry(SXF::OStream& out) 
  : SXF::ElemBucketRegistry(out)
{
  allocateRegistry();

  m_pErrorBucket = new UAL::SXFElemError(out);

  bind("entry", new UAL::SXFElemMltEntry(out));
  bind("exit",  new UAL::SXFElemMltExit(out));
  bind("align", new UAL::SXFElemAlign(out));
  bind("aperture", new UAL::SXFElemAperture(out));  //make the aperture round by default
}

// Destructor.
UAL::SXFElemBucketRegistry::~SXFElemBucketRegistry() {
  if(m_aBuckets){
    for(int i=0; i < m_iSize; i++){ 
      if(m_aBuckets[i]) { delete m_aBuckets[i]; }
    }  
    delete [] m_aBuckets;
  }
  if(m_pErrorBucket) { delete m_pErrorBucket; }
}

// Return singleton.
UAL::SXFElemBucketRegistry* UAL::SXFElemBucketRegistry::getInstance(SXF::OStream& out)
{
  if(!s_pBucketRegistry) { 
    s_pBucketRegistry = new UAL::SXFElemBucketRegistry(out);
  }
  return s_pBucketRegistry;
}

// Write data.
void UAL::SXFElemBucketRegistry::write(ostream& out, const PacLattElement& element, const string& tab)
{
  UAL::SXFElemBucket* writer;
  for(int i=0; i < m_iSize; i++){ 
    writer = (UAL::SXFElemBucket*) m_aBuckets[i];
    if(writer) writer->write(out, element, tab);  
  }  
}
