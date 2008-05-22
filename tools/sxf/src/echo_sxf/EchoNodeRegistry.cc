#include "echo_sxf/EchoElemBucket.hh"
#include "echo_sxf/EchoElement.hh"
#include "echo_sxf/EchoSequence.hh"
#include "echo_sxf/EchoNodeRegistry.hh"

#include "echo_sxf/EchoError.hh"

SXF::EchoNodeRegistry* SXF::EchoNodeRegistry::s_pNodeRegistry = 0;

// Constructor.
SXF::EchoNodeRegistry::EchoNodeRegistry(SXF::OStream& out)
  : SXF::NodeRegistry(out)
{
  allocateRegistry();

  SXF::EchoElemBucket* body;

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemEmptyHash());
  bind("marker",     new SXF::EchoElement(out, "marker", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemEmptyHash());
  bind("drift",      new SXF::EchoElement(out, "drift", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemBendHash());
  bind("rbend",      new SXF::EchoElement(out, "rbend", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemBendHash());
  bind("sbend",      new SXF::EchoElement(out, "sbend", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemMultipoleHash());
  bind("quadrupole", new SXF::EchoElement(out, "quadrupole", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemMultipoleHash());
  bind("sextupole",  new SXF::EchoElement(out, "sextupole", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemMultipoleHash());
  bind("octupole",   new SXF::EchoElement(out, "octupole", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemMultipoleHash());
  bind("multipole",  new SXF::EchoElement(out, "multipole", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemSolenoidHash());
  bind("solenoid",   new SXF::EchoElement(out, "solenoid", body)); 

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemMultipoleHash());
  bind("hkicker",    new SXF::EchoElement(out, "hkicker", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemMultipoleHash());
  bind("vkicker",    new SXF::EchoElement(out, "vkicker", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemMultipoleHash());
  bind("kicker",     new SXF::EchoElement(out, "kicker", body)); 

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemRfCavityHash());
  bind("rfcavity",   new SXF::EchoElement(out, "rfcavity", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemEmptyHash());
  bind("hmonitor",   new SXF::EchoElement(out, "hmonitor", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemEmptyHash());
  bind("vmonitor",   new SXF::EchoElement(out, "vmonitor", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemEmptyHash());
  bind("monitor",    new SXF::EchoElement(out, "monitor", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemCollimatorHash());
  bind("ecollimator",new SXF::EchoElement(out, "ecollimator", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemCollimatorHash());
  bind("rcollimator",new SXF::EchoElement(out, "rcollimator", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemElSeparatorHash());
  bind("elseparator",new SXF::EchoElement(out, "elseparator", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemEmptyHash());
  bind("instrument", new SXF::EchoElement(out, "instrument", body));

  body = new SXF::EchoElemBucket(out, "body", new SXF::ElemBeamBeamHash());
  bind("beambeam",   new SXF::EchoElement(out, "beambeam", body));


  m_pErrorElement =  new  SXF::EchoError(out, "error"); 
  m_pSequence     =  new  SXF::EchoSequence(out);  

}

// Destructor.
SXF::EchoNodeRegistry::~EchoNodeRegistry() 
{
  if(m_aElements){
    for(int i=0; i < m_iSize; i++){ 
      if(m_aElements[i]) { delete m_aElements[i]; }
    }  
    delete [] m_aElements;
  }
  if(m_pErrorElement) { delete m_pErrorElement; }
  if(m_pSequence)     { delete m_pSequence; }
}

// Return singleton.
SXF::EchoNodeRegistry* SXF::EchoNodeRegistry::getInstance(SXF::OStream& out)
{
  if(!s_pNodeRegistry) { 
    s_pNodeRegistry = new SXF::EchoNodeRegistry(out);
  }
  return s_pNodeRegistry;
}
