#include "UAL/SXF/buckets/ElemEmpty.hh"
#include "UAL/SXF/buckets/ElemBend.hh"
#include "UAL/SXF/buckets/ElemMltBody.hh"
#include "UAL/SXF/buckets/ElemSolenoid.hh"
#include "UAL/SXF/buckets/ElemKicker.hh"
#include "UAL/SXF/buckets/ElemRfCavity.hh"
#include "UAL/SXF/buckets/ElemCollimator.hh"

#include "UAL/SXF/Error.hh"
#include "UAL/SXF/Sequence.hh"
#include "UAL/SXF/NodeRegistry.hh"

UAL::SXFNodeRegistry* UAL::SXFNodeRegistry::s_pNodeRegistry = 0;

// Constructor.
UAL::SXFNodeRegistry::SXFNodeRegistry(SXF::OStream& out, PacSmf& smf)
  : SXF::NodeRegistry(out)
{
  allocateRegistry();

  UAL::SXFElemBucket* body;

  body = new UAL::SXFElemEmpty(out, "body");
  bind("marker",     new UAL::SXFElement(out, "marker", body, smf));

  body = new UAL::SXFElemEmpty(out, "body");
  bind("drift",      new UAL::SXFElement(out, "drift", body, smf));

  body = new UAL::SXFElemBend(out);
  bind("rbend",      new UAL::SXFElement(out, "rbend", body, smf));

  body = new UAL::SXFElemBend(out);
  bind("sbend",      new UAL::SXFElement(out, "sbend", body, smf));

  body = new UAL::SXFElemMltBody(out);
  bind("quadrupole", new UAL::SXFElement(out, "quadrupole", body, smf));

  body = new UAL::SXFElemMltBody(out);
  bind("sextupole",  new UAL::SXFElement(out, "sextupole", body, smf));

  body = new UAL::SXFElemMltBody(out);
  bind("octupole",   new UAL::SXFElement(out, "octupole", body, smf));

  body = new UAL::SXFElemMltBody(out);
  bind("multipole",  new UAL::SXFElement(out, "multipole", body, smf));

  body = new UAL::SXFElemSolenoid(out);
  bind("solenoid",   new UAL::SXFElement(out, "solenoid", body, smf)); 

  body = new UAL::SXFElemKicker(out);
  bind("hkicker",    new UAL::SXFElement(out, "hkicker", body, smf));

  body = new UAL::SXFElemKicker(out);
  bind("vkicker",    new UAL::SXFElement(out, "vkicker", body, smf));

  body = new UAL::SXFElemKicker(out);
  bind("kicker",     new UAL::SXFElement(out, "kicker", body, smf)); 

  body = new UAL::SXFElemRfCavity(out);
  bind("rfcavity",   new UAL::SXFElement(out, "rfcavity", body, smf));

  body = new UAL::SXFElemEmpty(out, "body");
  bind("hmonitor",   new UAL::SXFElement(out, "hmonitor", body, smf));

  body = new UAL::SXFElemEmpty(out, "body");
  bind("vmonitor",   new UAL::SXFElement(out, "vmonitor", body, smf));

  body = new UAL::SXFElemEmpty(out, "body");
  bind("monitor",    new UAL::SXFElement(out, "monitor", body, smf));

  body = new UAL::SXFElemCollimator(out, "body", 'e');
  bind("ecollimator",new UAL::SXFElement(out, "ecollimator", body, smf));

  body = new UAL::SXFElemCollimator(out, "body", 'r');
  bind("rcollimator",new UAL::SXFElement(out, "rcollimator", body, smf));

  body = new UAL::SXFElemEmpty(out, "body");
  bind("elseparator",new UAL::SXFElement(out, "drift", body, smf));

  body = new UAL::SXFElemEmpty(out, "body");
  bind("instrument", new UAL::SXFElement(out, "instrument", body, smf));

  body = new UAL::SXFElemEmpty(out, "body");
  bind("beambeam",   new UAL::SXFElement(out, "beambeam", body, smf));

  body = new UAL::SXFElemEmpty(out, "body");
  m_pErrorElement =  new  UAL::SXFError(out, "error", body, smf); 

  m_pSequence     =  new  UAL::SXFSequence(out);  

}

// Destructor.
UAL::SXFNodeRegistry::~SXFNodeRegistry() 
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
UAL::SXFNodeRegistry* 
UAL::SXFNodeRegistry::getInstance(SXF::OStream& out, PacSmf& smf)
{
  if(!s_pNodeRegistry) { 
    s_pNodeRegistry = new UAL::SXFNodeRegistry(out, smf);
  }
  return s_pNodeRegistry;
}
