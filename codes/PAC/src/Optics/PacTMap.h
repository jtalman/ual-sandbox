// Program     : Pac
// File        : Optics/PacTMap.h
// Description : Taylor Map
// Copyright   : see Copyright file
// Authors     : Nikolay Malitsky

#ifndef PAC_TAYLOR_MAP_H
#define PAC_TAYLOR_MAP_H

#include "UAL/Common/Probe.hh"
#include "PAC/Beam/BeamAttributes.hh"
#include "PAC/Beam/Bunch.hh"
#include "Optics/PacVTps.h"

/** 
 * Operates with Tailor Maps and Vectors of Truncated Power Series.
 */


class PacTMap : public PacVTps
{
public:

  // Constructors & destructor

  /** Constructor. Creates a PacTMap instance. The integer variable $size is 
      the phase-space dimension. */
  PacTMap(int size) : PacVTps() { create(size); }

  /** Copy constructor. Copies data from PacVTp into the new a PacTMap instance. */
  PacTMap(const PacVTps& vtps) : PacVTps(vtps) {}

  // Assignment operators

  /** Copy operator.*/
  PacTMap& operator  =(const PacVTps& vtps) { PacVTps::operator  =(vtps); return *this; }

  /** Returns beam attributes */
  PAC::BeamAttributes& getBeamAttributes() { return *this;}

  /** Returns the reference to the PacPosition object containing coordinates 
      of the reference orbit.*/
  PAC::Position refOrbit() const;

  /** Sets coordinates of the reference orbit by using the PacPosition object 
      containing new coordinates. */
  void refOrbit(const PAC::Position& p);  

  // Commands

  /** Propagates coordinates, contained in the PacPosition object, through the Tailor Map.*/
  void propagate(PAC::Position& p);

  /** Propagates the Vector of Truncated Power Series, contained in the PacVTps object, 
      through the Tailor Map. */
  void propagate(PacVTps& vtps);

  /** Propagates the Vector of Truncated Power Series, contained in the ZLIB::VTps object, 
      through the Tailor Map.  */
  void propagate(ZLIB::VTps& vtps);

  /** Propagates coordinates, contained in the PacPosition object, through the Tailor Map 
      "turns" times. */
  void propagate(PAC::Position& p, int turns);

  /** Propagates particles, contained in the PacBunch object, through the Tailor Map 
      "turns" times. */
  void propagate(PAC::Bunch& bunch, int turns = 1);

protected:

  void create(int size);
  void check(PAC::Particle& p, int ip, int turn);

private:

  PacTMap& operator  =(const PacTMap& ) { return *this; }  

};

#endif
