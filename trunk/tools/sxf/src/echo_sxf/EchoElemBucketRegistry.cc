#include "echo_sxf/EchoElemBucketRegistry.hh"
#include "echo_sxf/EchoElemError.hh"

SXF::EchoElemBucketRegistry* SXF::EchoElemBucketRegistry::s_pBucketRegistry = 0;

// Constructor
SXF::EchoElemBucketRegistry::EchoElemBucketRegistry(SXF::OStream& out) 
  : SXF::ElemBucketRegistry(out)
{
  allocateRegistry();

  m_pErrorBucket = new SXF::EchoElemError(out);

  bind("entry", new SXF::EchoElemBucket(out, "entry", new SXF::ElemMultipoleHash()));
  bind("exit",  new SXF::EchoElemBucket(out, "exit", new SXF::ElemMultipoleHash()));
  bind("align", new SXF::EchoElemBucket(out, "align", new SXF::ElemAlignHash()));
  bind("aperture",new SXF::EchoElemBucket(out, "aperture", new SXF::ElemApertureHash()));
}

// Destructor
SXF::EchoElemBucketRegistry::~EchoElemBucketRegistry() {
  if(m_aBuckets){
    for(int i=0; i < m_iSize; i++){ 
      if(m_aBuckets[i]) { delete m_aBuckets[i]; }
    }  
    delete [] m_aBuckets;
  }
  if(m_pErrorBucket) { delete m_pErrorBucket; }
}

// Return singleton
SXF::EchoElemBucketRegistry* SXF::EchoElemBucketRegistry::getInstance(OStream& out)
{
  if(!s_pBucketRegistry) { 
    s_pBucketRegistry = new SXF::EchoElemBucketRegistry(out);
  }
  return s_pBucketRegistry;
}


