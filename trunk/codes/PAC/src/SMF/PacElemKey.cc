// Library     : PAC
// File        : SMF/PacElemKey.cc
// Copyright   : see Copyright file
// Description : Implementation of PacElemKey
// Author      : Nikolay Malitsky

#include "SMF/PacElemKeys.h"

std::string PacElemKey::s_notype = "";

PacElemKey::PacElemKey(const std::string& name, int key)
  : _ptr(new PacElemKey::Data())
{
  _ptr->_key = key;
  _ptr->_name = name;

  if(!PacElemKeys::instance()->insert(*this)) { 
    std::string msg = "Error : PacElemKey(const String& name, int key) : insertion failed for ";
    PacDomainError(msg + name).raise();
  }
}

PacElemKey* PacElemKey::operator()(int key) const
{
  PacElemKeys::iterator i = PacElemKeys::instance()->find(key);
  if(i == PacElemKeys::instance()->end()) return 0;
  return &(*i);
}

void PacElemKey::checkName()
{ 
  if(!name().empty()){
    std::string msg = "Error: PacElemKey::checkName() : attempt to modify PacElemKey ";
    PacDomainError(msg + name()).raise();
  }
}

// PacKeyOfElemKey

void PacKeyOfElemKey::operator()(PacElemKey&, int) const
{
  std::string msg  = "Error : PacKeyOfElemKey::operator(PacElemKey& x, int key) const ";
         msg += ": don't insert items in collection \n";
  PacDomainError(msg).raise();
}

int PacKeyOfElemKey::count(const PacElemKey&) const
{
  std::string msg  = "Error : PacKeyOfElemKey::count(const PacElemKey& x ) const ";
         msg += ": don't erase items from collection \n";
  PacDomainError(msg).raise();
  return 0;
}


PacElemKey pacRbendKey("Rbend", 1);
PacElemKey pacSbendKey("Sbend", 2);
PacElemKey pacQuadrupoleKey("Quadrupole", 3);
PacElemKey pacSextupoleKey("Sextupole", 4);
PacElemKey pacOctupoleKey("Octupole", 5);
PacElemKey pacMultipoleKey("Multipole", 6);
PacElemKey pacSolenoidKey("Solenoid", 7);
PacElemKey pacHmonitorKey("Hmonitor", 8);
PacElemKey pacVmonitorKey("Vmonitor", 9);
PacElemKey pacMonitorKey("Monitor", 10);
PacElemKey pacHkickerKey("Hkicker", 11);
PacElemKey pacVkickerKey("Vkicker", 12);
PacElemKey pacKickerKey("Kicker", 13);
PacElemKey pacRfCavityKey("RfCavity", 14);
PacElemKey pacElSeparatorKey("ElSeparator", 15);
PacElemKey pacEcollimatorKey("Ecollimator", 16);
PacElemKey pacRcollimatorKey("Rcollimator", 17);
PacElemKey pacYrotKey("Yrot", 18);
PacElemKey pacSrotKey("Srot", 19);
PacElemKey pacInstrumentKey("Instrument", 20);
PacElemKey pacBeamBeamKey("BeamBeam", 21);
PacElemKey pacDriftKey("Drift", 97);
PacElemKey pacMarkerKey("Marker", 98);
PacElemKey pacElementKey("Element", 99);
