// Library       : SPINK
// File          : SPINK/SpinMatrix/SpinMapperFactory.cc
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#include "UAL/APF/PropagatorFactory.hh"
#include "SPINK/SpinMapper/SpinMapperFactory.hh"

SPINK::SpinMapper* SPINK::SpinMapperFactory::createDefaultMapper()
{
  return new SPINK::DriftSpinMapper();
}

SPINK::DriftSpinMapper* SPINK::SpinMapperFactory::createDriftMapper()
{
  return new SPINK::DriftSpinMapper();
}

/*
SPINK::MagnetSpinMapper* SPINK::SpinMapperFactory::createMagnetMapper()
{
  return new SPINK::MagnetSpinMapper();
}
*/

SPINK::SpinMapperRegister::SpinMapperRegister()
{
  UAL::PropagatorNodePtr driftPtr(new SPINK::DriftSpinMapper());
  UAL::PropagatorFactory::getInstance().add("SPINK::DriftSpinMapper", driftPtr);
}

static SPINK::SpinMapperRegister theSpinkSpinMapperRegister; 
