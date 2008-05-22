// Library       : TEAPOT
// File          : TEAPOT/Integrator/DaIntegratorFactory.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include "UAL/APF/PropagatorFactory.hh"
#include "TEAPOT/Integrator/DaIntegratorFactory.hh"

TEAPOT::BasicDaIntegrator* TEAPOT::DaIntegratorFactory::createDefaultDaIntegrator()
{
  return new TEAPOT::DriftDaIntegrator();
}

TEAPOT::DriftDaIntegrator* TEAPOT::DaIntegratorFactory::createDriftDaIntegrator()
{
  return new TEAPOT::DriftDaIntegrator();
}

TEAPOT::DipoleDaIntegrator* TEAPOT::DaIntegratorFactory::createDipoleDaIntegrator()
{
  return new TEAPOT::DipoleDaIntegrator();
}

TEAPOT::MltDaIntegrator* TEAPOT::DaIntegratorFactory::createMltDaIntegrator()
{
  return new TEAPOT::MltDaIntegrator();
}

TEAPOT::MapDaIntegrator* TEAPOT::DaIntegratorFactory::createMapDaIntegrator()
{
  return new TEAPOT::MapDaIntegrator();
}

TEAPOT::DaIntegratorRegister::DaIntegratorRegister()
{
  UAL::PropagatorNodePtr driftPtr(new TEAPOT::DriftDaIntegrator());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::DriftDaIntegrator", driftPtr);

  UAL::PropagatorNodePtr dipolePtr(new TEAPOT::DipoleDaIntegrator());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::DipoleDaIntegrator", dipolePtr);

  UAL::PropagatorNodePtr mltPtr(new TEAPOT::MltDaIntegrator());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::MltDaIntegrator", mltPtr);

  UAL::PropagatorNodePtr mapPtr(new TEAPOT::MapDaIntegrator());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::MapDaIntegrator", mapPtr);

}

static TEAPOT::DaIntegratorRegister theSingleton; 
