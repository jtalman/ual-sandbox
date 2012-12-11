#include "UAL/APF/PropagatorFactory.hh"
#include "ETEAPOT_MltTurn/Integrator/TrackerFactory.hh"

ETEAPOT_MltTurn::TrackerFactory* ETEAPOT_MltTurn::TrackerFactory::s_theInstance = 0;

ETEAPOT_MltTurn::TrackerFactory::TrackerFactory()
{
  UAL::PropagatorNodePtr dipolePtr(new ETEAPOT_MltTurn::DipoleTracker());
  m_trackers["Sbend"] = dipolePtr;
  m_trackers["Rbend"] = dipolePtr;

  UAL::PropagatorNodePtr mltPtr(new ETEAPOT_MltTurn::MltTracker());
  m_trackers["Kicker"]     = mltPtr;
  m_trackers["Hkicker"]    = mltPtr;  
  m_trackers["Vkicker"]    = mltPtr;
  m_trackers["Quadrupole"] = mltPtr;
  m_trackers["Sextupole"]  = mltPtr;  
  m_trackers["Multipole"]  = mltPtr;

  UAL::PropagatorNodePtr rfPtr(new ETEAPOT_MltTurn::RFCavityTracker());
  m_trackers["RfCavity"]   = rfPtr;

}

ETEAPOT_MltTurn::TrackerFactory* ETEAPOT_MltTurn::TrackerFactory::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new ETEAPOT_MltTurn::TrackerFactory();
  }
  return s_theInstance;
}

UAL::PropagatorNode* ETEAPOT_MltTurn::TrackerFactory::createTracker(const std::string& type)
{
  ETEAPOT_MltTurn::TrackerFactory* factory = getInstance(); 

  std::map<std::string, UAL::PropagatorNodePtr>::const_iterator it = factory->m_trackers.find(type);
  if(it == factory->m_trackers.end()) return createDefaultTracker();
  return it->second->clone();  
}

ETEAPOT::BasicTracker* ETEAPOT_MltTurn::TrackerFactory::createDefaultTracker()
{
  return new ETEAPOT_MltTurn::DriftTracker();
}

ETEAPOT_MltTurn::MarkerTracker* ETEAPOT_MltTurn::TrackerFactory::createMarkerTracker()
{
  return new ETEAPOT_MltTurn::MarkerTracker();
}

ETEAPOT_MltTurn::DriftTracker* ETEAPOT_MltTurn::TrackerFactory::createDriftTracker()
{
  return new ETEAPOT_MltTurn::DriftTracker();
}

ETEAPOT_MltTurn::DipoleTracker* ETEAPOT_MltTurn::TrackerFactory::createDipoleTracker()
{
  return new ETEAPOT_MltTurn::DipoleTracker();
}

ETEAPOT_MltTurn::MltTracker* ETEAPOT_MltTurn::TrackerFactory::createMltTracker()
{
  return new ETEAPOT_MltTurn::MltTracker();
}

// ETEAPOT_MltTurn::MatrixTracker* ETEAPOT_MltTurn::TrackerFactory::createMatrixTracker()
// {
//  return new ETEAPOT_MltTurn::MatrixTracker();
// }




ETEAPOT_MltTurn::TrackerRegister::TrackerRegister()
{
  UAL::PropagatorNodePtr markerPtr(new ETEAPOT_MltTurn::MarkerTracker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT_MltTurn::MarkerTracker", markerPtr);

  UAL::PropagatorNodePtr driftPtr(new ETEAPOT_MltTurn::DriftTracker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT_MltTurn::DriftTracker", driftPtr);

  UAL::PropagatorNodePtr dipolePtr(new ETEAPOT_MltTurn::DipoleTracker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT_MltTurn::DipoleTracker", dipolePtr);

  UAL::PropagatorNodePtr mltPtr(new ETEAPOT_MltTurn::MltTracker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT_MltTurn::MltTracker", mltPtr);

  // UAL::PropagatorNodePtr matrixPtr(new ETEAPOT_MltTurn::MatrixTracker());
  // UAL::PropagatorFactory::getInstance().add("ETEAPOT_MltTurn::MatrixTracker", matrixPtr);

  UAL::PropagatorNodePtr rfPtr(new ETEAPOT_MltTurn::RFCavityTracker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT_MltTurn::RFCavityTracker", rfPtr);
}

static ETEAPOT_MltTurn::TrackerRegister theSingleton; 
