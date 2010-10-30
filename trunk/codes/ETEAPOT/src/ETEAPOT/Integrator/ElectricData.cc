// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/ElectricData.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>

#include "SMF/PacElemAttributes.h"
#include "ETEAPOT/Integrator/ElectricData.hh"

ETEAPOT::ElectricData::ElectricData()
{
  initialize();
}

ETEAPOT::ElectricData::ElectricData(const ElectricData& edata)
{
  copy(edata);
}


ETEAPOT::ElectricData::~ElectricData()
{
}

const ETEAPOT::ElectricData& ETEAPOT::ElectricData::operator=(const ElectricData& edata)
{
  copy(edata);
  return *this;
}

void ETEAPOT::ElectricData::setLatticeElement(const PacLattElement& e)
{

  // Entry multipole
  PacElemAttributes* front  = e.getFront();
  if(front){
     PacElemAttributes::iterator it = front->find(PAC_MULTIPOLE);
     if(it != front->end()) m_entryMlt = (PacElemMultipole*) &(*it);
  }

  // Exit multipole
  PacElemAttributes* end  = e.getEnd();
  if(end){
     PacElemAttributes::iterator it = end->find(PAC_MULTIPOLE);
     if(it != end->end()) m_exitMlt = (PacElemMultipole*) &(*it);
  }

  // Body attributes
  PacElemAttributes* attributes = e.getBody();

  if(attributes){
    for(PacElemAttributes::iterator it = attributes->begin(); it != attributes->end(); it++){
      switch((*it).key()){
      case PAC_MULTIPOLE:
	m_mlt = (PacElemMultipole*) &(*it);
	break;
      case PAC_OFFSET:
	m_offset = (PacElemOffset*) &(*it);
	break;
      case PAC_APERTURE:
	// m_aperture = (PacElemAperture*) &(*it);
	break;
      case PAC_ROTATION:
	m_rotation = (PacElemRotation*) &(*it);
	break;
      default:
	break;
      }
    }
  } 
}

void ETEAPOT::ElectricData::initialize()
{

  m_entryMlt = 0;
  m_exitMlt = 0;

  m_mlt = 0;
  m_offset = 0;
  m_rotation = 0;
  // m_aperture = 0;
}

void ETEAPOT::ElectricData::copy(const ETEAPOT::ElectricData& edata)
{

  m_entryMlt = edata.m_entryMlt;
  m_exitMlt = edata.m_exitMlt;

  m_mlt = edata.m_mlt;
  m_offset = edata.m_offset;
  m_rotation = edata.m_rotation;
  // m_aperture = edata.m_aperture;
}
