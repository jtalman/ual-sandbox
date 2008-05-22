// Library       : TEAPOT
// File          : TEAPOT/Integrator/MagnetData.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>

#include "SMF/PacElemAttributes.h"
#include "TEAPOT/Integrator/MagnetData.hh"

TEAPOT::MagnetData::MagnetData()
{
  initialize();
}

TEAPOT::MagnetData::MagnetData(const MagnetData& mdata)
{
  copy(mdata);
}


TEAPOT::MagnetData::~MagnetData()
{
}

const TEAPOT::MagnetData& TEAPOT::MagnetData::operator=(const MagnetData& mdata)
{
  copy(mdata);
  return *this;
}

void TEAPOT::MagnetData::setLatticeElement(const PacLattElement& e)
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

void TEAPOT::MagnetData::initialize()
{

  m_entryMlt = 0;
  m_exitMlt = 0;

  m_mlt = 0;
  m_offset = 0;
  m_rotation = 0;
  // m_aperture = 0;
}

void TEAPOT::MagnetData::copy(const TEAPOT::MagnetData& mdata)
{

  m_entryMlt = mdata.m_entryMlt;
  m_exitMlt = mdata.m_exitMlt;

  m_mlt = mdata.m_mlt;
  m_offset = mdata.m_offset;
  m_rotation = mdata.m_rotation;
  // m_aperture = mdata.m_aperture;
}
