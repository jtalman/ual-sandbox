// Library     : PAC
// File        : SMF/PacElemKey.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_KEY_H
#define PAC_ELEM_KEY_H

#include "Templates/PacRCIPtr.h"

class PacElemKey
{

public:

// Constructors & copy operator

  PacElemKey() : _ptr(new PacElemKey::Data()) {}
  PacElemKey(const std::string& name, int key);
  PacElemKey(const PacElemKey& key) : _ptr(key._ptr) {}

  PacElemKey& operator = (const PacElemKey& key) { checkName(); _ptr = key._ptr; return *this; }

// Key & Count

   const int& key() const { return _ptr->_key; }
   int count() const { return _ptr.count(); }

// Data

   const std::string& name() const { return _ptr->_name; }

// Interface to global items

   PacElemKey* operator()(int key) const;

   static std::string s_notype;

protected:

  class Data
  {
  public:

    Data() : _key(0) {}

    int _key;
    std::string _name;
  };

  typedef PacRCIPtr<Data> smart_pointer;
  smart_pointer _ptr;

private:

  void checkName();

};

struct PacKeyOfElemKey
{
  const int& operator()(const PacElemKey& x) const { return x.key(); }

  // it is prohibited to prevent the insertion of new PacElemKey 
  void operator()(PacElemKey& x, int key) const; 

  // it is prohibited to prevent the erasion of PacElemKey
  int count(const PacElemKey& x) const ; 
};


extern PacElemKey pacRbendKey;       // pacRbendKey("Rbend", 1);
extern PacElemKey pacSbendKey;       // pacSbendKey("Sbend", 2);
extern PacElemKey pacQuadrupoleKey;  // pacQuadrupoleKey("Quadrupole", 3);
extern PacElemKey pacSextupoleKey;   // pacSextupoleKey("Sextupole", 4);
extern PacElemKey pacOctupole;       // pacOctupoleKey("Octupole", 5);
extern PacElemKey pacMultipoleKey;   // pacMultipoleKey("Multipole", 6);
extern PacElemKey pacSolenoidKey;    // pacSolenoidKey("Solenoid", 7);
extern PacElemKey pacHmonitorKey;    // pacHmonitorKey("Hmonitor", 8);
extern PacElemKey pacVmonitorKey;    // pacVmonitorKey("Vmonitor", 9);
extern PacElemKey pacMonitorKey;     // pacMonitorKey("Monitor", 10);
extern PacElemKey pacHkickerKey;     // pacHkickerKey("Hkicker", 11);
extern PacElemKey pacVkickerKey;     // pacVkickerKey("Vkicker", 12);
extern PacElemKey pacKickerKey;      // pacKickerKey("Kicker", 13);
extern PacElemKey pacRfCavityKey;    // pacRfCavityKey("RfCavity", 14);
extern PacElemKey pacElSeparatorKey; // pacElSeparatorKey("ElSeparator", 15);
extern PacElemKey pacEcollimatorKey; // pacEcollimatorKey("Ecollimator", 16);
extern PacElemKey pacRcollimatorKey; // pacRcollimatorKey("Rcollimator", 17);
extern PacElemKey pacYrotKey;        // pacYrotKey("Yrot", 18);
extern PacElemKey pacSrotKey;        // pacSrotKey("Srot", 19);
extern PacElemKey pacInstrumentKey;  // pacInstrumentKey("Instrument", 20);
extern PacElemKey pacBeamBeamKey;    // pacBeamBeamKey("BeamBeam", 21);
extern PacElemKey pacDriftKey;       // pacDriftKey("Drift", 97);
extern PacElemKey pacMarkerKey;      // pacMarkerKey("Marker", 98);
extern PacElemKey pacElementKey;     // pacElementKey("Element", 99);

#endif
