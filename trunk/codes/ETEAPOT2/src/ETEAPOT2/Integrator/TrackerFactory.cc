#include "UAL/APF/PropagatorFactory.hh"
#include "ETEAPOT2/Integrator/TrackerFactory.hh"

#include"ETEAPOT2/Integrator/genMethods/spinExtern"

ETEAPOT2::TrackerFactory* ETEAPOT2::TrackerFactory::s_theInstance = 0;

int ETEAPOT2::marker::turns=0;
int ETEAPOT2::marker::markerCount=0;
double ETEAPOT2::bend::dZFF;
double ETEAPOT2::bend::m_m;
int ETEAPOT2::bend::bnd=0;
//int ETEAPOT2::bend::bndsPerTrn;
//double ETEAPOT2::bend::spin[41][3];

ETEAPOT2::TrackerFactory::TrackerFactory()
{
  UAL::PropagatorNodePtr dipolePtr(new ETEAPOT2::bend());
  m_trackers["Sbend"] = dipolePtr;
  m_trackers["Rbend"] = dipolePtr;

//UAL::PropagatorNodePtr mltPtr(new ETEAPOT2::mlt());
  UAL::PropagatorNodePtr quadPtr(new ETEAPOT2::quad());
  UAL::PropagatorNodePtr sextPtr(new ETEAPOT2::sext());
  UAL::PropagatorNodePtr octPtr(new ETEAPOT2::oct());
//m_trackers["Kicker"]     = mltPtr;
//m_trackers["Hkicker"]    = mltPtr;  
//m_trackers["Vkicker"]    = mltPtr;
  m_trackers["Quadrupole"] = quadPtr;
  m_trackers["Sextupole"]  = sextPtr;  
  m_trackers["Octupole"]  = octPtr;  
//m_trackers["Multipole"]  = mltPtr;

  UAL::PropagatorNodePtr rfPtr(new ETEAPOT2::rfCavity());
  m_trackers["RfCavity"]   = rfPtr;

}

ETEAPOT2::TrackerFactory* ETEAPOT2::TrackerFactory::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new ETEAPOT2::TrackerFactory();
  }
  return s_theInstance;
}

UAL::PropagatorNode* ETEAPOT2::TrackerFactory::createTracker(const std::string& type)
{
  ETEAPOT2::TrackerFactory* factory = getInstance(); 

  std::map<std::string, UAL::PropagatorNodePtr>::const_iterator it = factory->m_trackers.find(type);
  if(it == factory->m_trackers.end()) return createDefaultTracker();
  return it->second->clone();  
}

ETEAPOT::BasicTracker* ETEAPOT2::TrackerFactory::createDefaultTracker()
{
  return new ETEAPOT2::drift();
}

ETEAPOT2::marker* ETEAPOT2::TrackerFactory::createmarker()
{
  return new ETEAPOT2::marker();
}

ETEAPOT2::drift* ETEAPOT2::TrackerFactory::createdrift()
{
  return new ETEAPOT2::drift();
}

ETEAPOT2::bend* ETEAPOT2::TrackerFactory::createbend()
{
  return new ETEAPOT2::bend();
}

ETEAPOT2::quad* ETEAPOT2::TrackerFactory::createquad()
{
  return new ETEAPOT2::quad();
}

ETEAPOT2::sext* ETEAPOT2::TrackerFactory::createsext()
{
  return new ETEAPOT2::sext();
}

ETEAPOT2::oct* ETEAPOT2::TrackerFactory::createoct()
{
  return new ETEAPOT2::oct();
}

ETEAPOT2::rfCavity* ETEAPOT2::TrackerFactory::createrfCavity()
{
  return new ETEAPOT2::rfCavity();
}

// ETEAPOT2::MatrixTracker* ETEAPOT2::TrackerFactory::createMatrixTracker()
// {
//  return new ETEAPOT2::MatrixTracker();
// }




ETEAPOT2::TrackerRegister::TrackerRegister()
{
  UAL::PropagatorNodePtr markerPtr(new ETEAPOT2::marker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT2::marker", markerPtr);

  UAL::PropagatorNodePtr driftPtr(new ETEAPOT2::drift());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT2::drift", driftPtr);

  UAL::PropagatorNodePtr dipolePtr(new ETEAPOT2::bend());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT2::bend", dipolePtr);

  UAL::PropagatorNodePtr quadPtr(new ETEAPOT2::quad());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT2::quad", quadPtr);

  UAL::PropagatorNodePtr sextPtr(new ETEAPOT2::sext());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT2::sext", sextPtr);

  UAL::PropagatorNodePtr octPtr(new ETEAPOT2::oct());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT2::oct", octPtr);

  // UAL::PropagatorNodePtr matrixPtr(new ETEAPOT2::MatrixTracker());
  // UAL::PropagatorFactory::getInstance().add("ETEAPOT2::MatrixTracker", matrixPtr);

  UAL::PropagatorNodePtr rfPtr(new ETEAPOT2::rfCavity());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT2::rfCavity", rfPtr);
}

static ETEAPOT2::TrackerRegister theSingleton; 
