#include "UAL/APF/PropagatorFactory.hh"
#include "EMTEAPOT/Integrator/TrackerFactory.hh"

#include"EMTEAPOT/Integrator/genMethods/spinExtern"

EMTEAPOT::TrackerFactory* EMTEAPOT::TrackerFactory::s_theInstance = 0;

int EMTEAPOT::marker::turns=0;
int EMTEAPOT::marker::markerCount=0;
double EMTEAPOT::embend::dZFF;
double EMTEAPOT::embend::m_m;
int EMTEAPOT::embend::bnd=0;

EMTEAPOT::TrackerFactory::TrackerFactory()
{
  UAL::PropagatorNodePtr dipolePtr(new EMTEAPOT::embend());
  m_trackers["Sbend"] = dipolePtr;
  m_trackers["Rbend"] = dipolePtr;

  UAL::PropagatorNodePtr quadPtr(new EMTEAPOT::quad());
  UAL::PropagatorNodePtr sextPtr(new EMTEAPOT::sext());
  UAL::PropagatorNodePtr octPtr(new EMTEAPOT::oct());
  m_trackers["Quadrupole"] = quadPtr;
  m_trackers["Sextupole"]  = sextPtr;  
  m_trackers["Octupole"]  = octPtr;  

  UAL::PropagatorNodePtr rfPtr(new EMTEAPOT::rfCavity());
  m_trackers["RfCavity"]   = rfPtr;

}

EMTEAPOT::TrackerFactory* EMTEAPOT::TrackerFactory::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new EMTEAPOT::TrackerFactory();
  }
  return s_theInstance;
}

UAL::PropagatorNode* EMTEAPOT::TrackerFactory::createTracker(const std::string& type)
{
  EMTEAPOT::TrackerFactory* factory = getInstance(); 

  std::map<std::string, UAL::PropagatorNodePtr>::const_iterator it = factory->m_trackers.find(type);
  if(it == factory->m_trackers.end()) return createDefaultTracker();
  return it->second->clone();  
}

ETEAPOT::BasicTracker* EMTEAPOT::TrackerFactory::createDefaultTracker()
{
  return new EMTEAPOT::drift();
}

EMTEAPOT::marker* EMTEAPOT::TrackerFactory::createmarker()
{
  return new EMTEAPOT::marker();
}

EMTEAPOT::drift* EMTEAPOT::TrackerFactory::createdrift()
{
  return new EMTEAPOT::drift();
}

EMTEAPOT::embend* EMTEAPOT::TrackerFactory::createembend()
{
  return new EMTEAPOT::embend();
}

EMTEAPOT::quad* EMTEAPOT::TrackerFactory::createquad()
{
  return new EMTEAPOT::quad();
}

EMTEAPOT::sext* EMTEAPOT::TrackerFactory::createsext()
{
  return new EMTEAPOT::sext();
}

EMTEAPOT::oct* EMTEAPOT::TrackerFactory::createoct()
{
  return new EMTEAPOT::oct();
}

EMTEAPOT::rfCavity* EMTEAPOT::TrackerFactory::createrfCavity()
{
  return new EMTEAPOT::rfCavity();
}

EMTEAPOT::TrackerRegister::TrackerRegister()
{
  UAL::PropagatorNodePtr markerPtr(new EMTEAPOT::marker());
  UAL::PropagatorFactory::getInstance().add("EMTEAPOT::marker", markerPtr);

  UAL::PropagatorNodePtr driftPtr(new EMTEAPOT::drift());
  UAL::PropagatorFactory::getInstance().add("EMTEAPOT::drift", driftPtr);

  UAL::PropagatorNodePtr dipolePtr(new EMTEAPOT::embend());
  UAL::PropagatorFactory::getInstance().add("EMTEAPOT::embend", dipolePtr);

  UAL::PropagatorNodePtr quadPtr(new EMTEAPOT::quad());
  UAL::PropagatorFactory::getInstance().add("EMTEAPOT::quad", quadPtr);

  UAL::PropagatorNodePtr sextPtr(new EMTEAPOT::sext());
  UAL::PropagatorFactory::getInstance().add("EMTEAPOT::sext", sextPtr);

  UAL::PropagatorNodePtr octPtr(new EMTEAPOT::oct());
  UAL::PropagatorFactory::getInstance().add("EMTEAPOT::oct", octPtr);

  UAL::PropagatorNodePtr rfPtr(new EMTEAPOT::rfCavity());
  UAL::PropagatorFactory::getInstance().add("EMTEAPOT::rfCavity", rfPtr);
}

static EMTEAPOT::TrackerRegister theSingleton; 
