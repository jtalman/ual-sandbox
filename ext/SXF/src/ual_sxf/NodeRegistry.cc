#include "ual_sxf/buckets/ElemEmpty.hh"
#include "ual_sxf/buckets/ElemBend.hh"
#include "ual_sxf/buckets/ElemMltBody.hh"
#include "ual_sxf/buckets/ElemSolenoid.hh"
#include "ual_sxf/buckets/ElemKicker.hh"
#include "ual_sxf/buckets/ElemRfCavity.hh"
#include "ual_sxf/buckets/ElemCollimator.hh"

#include "ual_sxf/Error.hh"
#include "ual_sxf/Sequence.hh"
#include "ual_sxf/NodeRegistry.hh"

UAL_SXF_NodeRegistry* UAL_SXF_NodeRegistry::s_pNodeRegistry = 0;

// Constructor.
UAL_SXF_NodeRegistry::UAL_SXF_NodeRegistry(SXF::OStream& out, PacSmf& smf)
  : SXF::NodeRegistry(out)
{
  allocateRegistry();

  UAL_SXF_ElemBucket* body;

  body = new UAL_SXF_ElemEmpty(out, "body");
  bind("marker",     new UAL_SXF_Element(out, "marker", body, smf));

  body = new UAL_SXF_ElemEmpty(out, "body");
  bind("drift",      new UAL_SXF_Element(out, "drift", body, smf));

  body = new UAL_SXF_ElemBend(out);
  bind("rbend",      new UAL_SXF_Element(out, "rbend", body, smf));

  body = new UAL_SXF_ElemBend(out);
  bind("sbend",      new UAL_SXF_Element(out, "sbend", body, smf));

  body = new UAL_SXF_ElemMltBody(out);
  bind("quadrupole", new UAL_SXF_Element(out, "quadrupole", body, smf));

  body = new UAL_SXF_ElemMltBody(out);
  bind("sextupole",  new UAL_SXF_Element(out, "sextupole", body, smf));

  body = new UAL_SXF_ElemMltBody(out);
  bind("octupole",   new UAL_SXF_Element(out, "octupole", body, smf));

  body = new UAL_SXF_ElemMltBody(out);
  bind("multipole",  new UAL_SXF_Element(out, "multipole", body, smf));

  body = new UAL_SXF_ElemSolenoid(out);
  bind("solenoid",   new UAL_SXF_Element(out, "solenoid", body, smf)); 

  body = new UAL_SXF_ElemKicker(out);
  bind("hkicker",    new UAL_SXF_Element(out, "hkicker", body, smf));

  body = new UAL_SXF_ElemKicker(out);
  bind("vkicker",    new UAL_SXF_Element(out, "vkicker", body, smf));

  body = new UAL_SXF_ElemKicker(out);
  bind("kicker",     new UAL_SXF_Element(out, "kicker", body, smf)); 

  body = new UAL_SXF_ElemRfCavity(out);
  bind("rfcavity",   new UAL_SXF_Element(out, "rfcavity", body, smf));

  body = new UAL_SXF_ElemEmpty(out, "body");
  bind("hmonitor",   new UAL_SXF_Element(out, "hmonitor", body, smf));

  body = new UAL_SXF_ElemEmpty(out, "body");
  bind("vmonitor",   new UAL_SXF_Element(out, "vmonitor", body, smf));

  body = new UAL_SXF_ElemEmpty(out, "body");
  bind("monitor",    new UAL_SXF_Element(out, "monitor", body, smf));

  body = new UAL_SXF_ElemCollimator(out, "body", 'e');
  bind("ecollimator",new UAL_SXF_Element(out, "ecollimator", body, smf));

  body = new UAL_SXF_ElemCollimator(out, "body", 'r');
  bind("rcollimator",new UAL_SXF_Element(out, "rcollimator", body, smf));

  body = new UAL_SXF_ElemEmpty(out, "body");
  bind("elseparator",new UAL_SXF_Element(out, "drift", body, smf));

  body = new UAL_SXF_ElemEmpty(out, "body");
  bind("instrument", new UAL_SXF_Element(out, "drift", body, smf));

  body = new UAL_SXF_ElemEmpty(out, "body");
  bind("beambeam",   new UAL_SXF_Element(out, "drift", body, smf));

  body = new UAL_SXF_ElemEmpty(out, "body");
  m_pErrorElement =  new  UAL_SXF_Error(out, "error", body, smf); 

  m_pSequence     =  new  UAL_SXF_Sequence(out);  

}

// Destructor.
UAL_SXF_NodeRegistry::~UAL_SXF_NodeRegistry() 
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
UAL_SXF_NodeRegistry* 
UAL_SXF_NodeRegistry::getInstance(SXF::OStream& out, PacSmf& smf)
{
  if(!s_pNodeRegistry) { 
    s_pNodeRegistry = new UAL_SXF_NodeRegistry(out, smf);
  }
  return s_pNodeRegistry;
}
