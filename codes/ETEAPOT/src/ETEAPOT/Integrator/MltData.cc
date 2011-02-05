// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/MltData.cc
// Copyright     : see Copyright file


#include <math.h>

#include "SMF/PacElemAttributes.h"
#include "ETEAPOT/Integrator/MltData.hh"

ETEAPOT::MltData::MltData()
{
  initialize();
}

ETEAPOT::MltData::MltData(const MltData& edata)
{
  copy(edata);
}


ETEAPOT::MltData::~MltData()
{
}

const ETEAPOT::MltData& ETEAPOT::MltData::operator=(const MltData& edata)
{
  copy(edata);
  return *this;
}

void ETEAPOT::MltData::setLatticeElement(const PacLattElement& e)
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

void ETEAPOT::MltData::initialize()
{

  m_entryMlt = 0;
  m_exitMlt = 0;

  m_mlt = 0;
  m_offset = 0;
  m_rotation = 0;
  // m_aperture = 0;
}

void ETEAPOT::MltData::copy(const ETEAPOT::MltData& edata)
{

  m_entryMlt = edata.m_entryMlt;
  m_exitMlt = edata.m_exitMlt;

  m_mlt = edata.m_mlt;
  m_offset = edata.m_offset;
  m_rotation = edata.m_rotation;
  // m_aperture = edata.m_aperture;
}
