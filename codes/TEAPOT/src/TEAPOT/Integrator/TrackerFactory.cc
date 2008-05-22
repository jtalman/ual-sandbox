// Library       : TEAPOT
// File          : TEAPOT/Integrator/TrackerFactory.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 


#include "UAL/APF/PropagatorFactory.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"

TEAPOT::TrackerFactory* TEAPOT::TrackerFactory::s_theInstance = 0;

TEAPOT::TrackerFactory::TrackerFactory()
{
  UAL::PropagatorNodePtr dipolePtr(new TEAPOT::DipoleTracker());
  m_trackers["Sbend"] = dipolePtr;
  m_trackers["Rbend"] = dipolePtr;

  UAL::PropagatorNodePtr mltPtr(new TEAPOT::MltTracker());
  m_trackers["Kicker"]     = mltPtr;
  m_trackers["Hkicker"]    = mltPtr;  
  m_trackers["Vkicker"]    = mltPtr;
  m_trackers["Quadrupole"] = mltPtr;
  m_trackers["Sextupole"]  = mltPtr;  
  m_trackers["Multipole"]  = mltPtr;

  UAL::PropagatorNodePtr rfPtr(new TEAPOT::RFCavityTracker());
  m_trackers["RfCavity"]   = rfPtr;

}

TEAPOT::TrackerFactory* TEAPOT::TrackerFactory::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new TEAPOT::TrackerFactory();
  }
  return s_theInstance;
}

UAL::PropagatorNode* TEAPOT::TrackerFactory::createTracker(const std::string& type)
{
  TEAPOT::TrackerFactory* factory = getInstance(); 

  std::map<std::string, UAL::PropagatorNodePtr>::const_iterator it = factory->m_trackers.find(type);
  if(it == factory->m_trackers.end()) return createDefaultTracker();
  return it->second->clone();  
}

TEAPOT::BasicTracker* TEAPOT::TrackerFactory::createDefaultTracker()
{
  return new TEAPOT::DriftTracker();
}

TEAPOT::DriftTracker* TEAPOT::TrackerFactory::createDriftTracker()
{
  return new TEAPOT::DriftTracker();
}

TEAPOT::DipoleTracker* TEAPOT::TrackerFactory::createDipoleTracker()
{
  return new TEAPOT::DipoleTracker();
}

TEAPOT::MltTracker* TEAPOT::TrackerFactory::createMltTracker()
{
  return new TEAPOT::MltTracker();
}

TEAPOT::MatrixTracker* TEAPOT::TrackerFactory::createMatrixTracker()
{
  return new TEAPOT::MatrixTracker();
}




TEAPOT::TrackerRegister::TrackerRegister()
{
  UAL::PropagatorNodePtr driftPtr(new TEAPOT::DriftTracker());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::DriftTracker", driftPtr);

  UAL::PropagatorNodePtr dipolePtr(new TEAPOT::DipoleTracker());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::DipoleTracker", dipolePtr);

  UAL::PropagatorNodePtr mltPtr(new TEAPOT::MltTracker());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::MltTracker", mltPtr);

  UAL::PropagatorNodePtr matrixPtr(new TEAPOT::MatrixTracker());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::MatrixTracker", matrixPtr);

  UAL::PropagatorNodePtr rfPtr(new TEAPOT::RFCavityTracker());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::RFCavityTracker", rfPtr);
}

static TEAPOT::TrackerRegister theSingleton; 
