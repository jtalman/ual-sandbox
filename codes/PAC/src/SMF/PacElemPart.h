// Library     : PAC
// File        : SMF/PacElemPart.h
// Copyright   : see Copyright file
// Description : Element part (body, front, end)
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_PART_H
#define PAC_ELEM_PART_H

#include "SMF/PacElemAttributes.h"

class PacElemPart
{

public:

  // Constructors & destructor

  PacElemPart(){}

  // "Standard Input Language"

  void set(const PacElemAttributes& att) { _value.set(att); }
  void set(const PacElemBucket& bucket)  { _value.set(bucket);}

  void add(const PacElemAttributes& att) { _value.add(att); }
  void add(const PacElemBucket& bucket)  { _value.add(bucket);} 

  double get(const PacElemAttribKey& key) const { return _value.get(key); }

  void remove(const PacElemAttribKey& key) { _value.remove(key); }
  void remove() { _value.remove(); }

  // Interface

  const PacElemAttributes& attributes() const { return _value; }
  PacElemAttributes& attributes() { return _value; }

  const PacElemAttributes& rms() const { return _rms; }
  PacElemAttributes& rms() { return _rms; }

private:

  PacElemAttributes  _value;
  PacElemAttributes  _rms;  

};

#endif
