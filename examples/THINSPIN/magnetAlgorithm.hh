// Library       : 
// File          : 
// Copyright     : see Copyright file
// Author        : 
// C++ version   : N.Malitsky and J.Talman

#ifndef THINSPIN_SPIN_MAGNET_ALGORITHM_HH
#define THINSPIN_SPIN_MAGNET_ALGORITHM_HH

#include "SMF/PacElemMultipole.h"
#include "TEAPOT/Integrator/MagnetData.hh"
#include "commonAlgorithm.hh"
#include "extern_globalBlock.cc"

namespace THINSPIN {
 
  /** A template of the common methods used by the magnet conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class magnetAlgorithm 
    : public commonAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    magnetAlgorithm();

    /** Destructor */
    ~magnetAlgorithm();

    /** Passes the element entry  */
    void passEntry(const TEAPOT::MagnetData& mdata, Coordinates& p); 

    /** Passes the element exit  */
    void passExit(const TEAPOT::MagnetData& mdata, Coordinates& p); 

   /** Applies the multipole kick */
   void applyMltKick(const TEAPOT::MagnetData& mdata, double rkicks, Coordinates& p);

  protected:
   
    /** Applies the multipole kick */
    void applyMltKick(PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p);

  };

}

#include "magnetAlgorithm.icc"

#endif
