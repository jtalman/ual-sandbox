// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/TrackerFactory.cc
// Copyright     : see Copyright file


#include "UAL/APF/PropagatorFactory.hh"
#include "ETEAPOT/Integrator/TrackerFactory.hh"

ETEAPOT::TrackerFactory* ETEAPOT::TrackerFactory::s_theInstance = 0;

ETEAPOT::TrackerFactory::TrackerFactory()
{
  UAL::PropagatorNodePtr dipolePtr(new ETEAPOT::DipoleTracker());
  m_trackers["Sbend"] = dipolePtr;
  m_trackers["Rbend"] = dipolePtr;

  UAL::PropagatorNodePtr mltPtr(new ETEAPOT::MltTracker());
  m_trackers["Kicker"]     = mltPtr;
  m_trackers["Hkicker"]    = mltPtr;  
  m_trackers["Vkicker"]    = mltPtr;
  m_trackers["Quadrupole"] = mltPtr;
  m_trackers["Sextupole"]  = mltPtr;  
  m_trackers["Multipole"]  = mltPtr;

  UAL::PropagatorNodePtr rfPtr(new ETEAPOT::RFCavityTracker());
  m_trackers["RfCavity"]   = rfPtr;

}

ETEAPOT::TrackerFactory* ETEAPOT::TrackerFactory::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new ETEAPOT::TrackerFactory();
  }
  return s_theInstance;
}

UAL::PropagatorNode* ETEAPOT::TrackerFactory::createTracker(const std::string& type)
{
  ETEAPOT::TrackerFactory* factory = getInstance(); 

  std::map<std::string, UAL::PropagatorNodePtr>::const_iterator it = factory->m_trackers.find(type);
  if(it == factory->m_trackers.end()) return createDefaultTracker();
  return it->second->clone();  
}

ETEAPOT::BasicTracker* ETEAPOT::TrackerFactory::createDefaultTracker()
{
  return new ETEAPOT::DriftTracker();
}

ETEAPOT::MarkerTracker* ETEAPOT::TrackerFactory::createMarkerTracker()
{
  return new ETEAPOT::MarkerTracker();
}

ETEAPOT::DriftTracker* ETEAPOT::TrackerFactory::createDriftTracker()
{
  return new ETEAPOT::DriftTracker();
}

ETEAPOT::DipoleTracker* ETEAPOT::TrackerFactory::createDipoleTracker()
{
  return new ETEAPOT::DipoleTracker();
}

ETEAPOT::MltTracker* ETEAPOT::TrackerFactory::createMltTracker()
{
  return new ETEAPOT::MltTracker();
}

// ETEAPOT::MatrixTracker* ETEAPOT::TrackerFactory::createMatrixTracker()
// {
//  return new ETEAPOT::MatrixTracker();
// }




ETEAPOT::TrackerRegister::TrackerRegister()
{
  UAL::PropagatorNodePtr markerPtr(new ETEAPOT::MarkerTracker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT::MarkerTracker", markerPtr);

  UAL::PropagatorNodePtr driftPtr(new ETEAPOT::DriftTracker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT::DriftTracker", driftPtr);

  UAL::PropagatorNodePtr dipolePtr(new ETEAPOT::DipoleTracker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT::DipoleTracker", dipolePtr);

  UAL::PropagatorNodePtr mltPtr(new ETEAPOT::MltTracker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT::MltTracker", mltPtr);

  // UAL::PropagatorNodePtr matrixPtr(new ETEAPOT::MatrixTracker());
  // UAL::PropagatorFactory::getInstance().add("ETEAPOT::MatrixTracker", matrixPtr);

  UAL::PropagatorNodePtr rfPtr(new ETEAPOT::RFCavityTracker());
  UAL::PropagatorFactory::getInstance().add("ETEAPOT::RFCavityTracker", rfPtr);
}

static ETEAPOT::TrackerRegister theSingleton; 
