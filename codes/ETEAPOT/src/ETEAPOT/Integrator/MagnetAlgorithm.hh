// Library       : TEAPOT
// File          : TEAPOT/Integrator/MagnetAlgorithm.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_MAGNET_ALGORITHM_HH
#define UAL_TEAPOT_MAGNET_ALGORITHM_HH

#include "SMF/PacElemMultipole.h"
#include "TEAPOT/Integrator/MagnetData.hh"
#include "TEAPOT/Integrator/CommonAlgorithm.hh"

namespace TEAPOT {
 
  /** A template of the common methods used by the magnet conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class MagnetAlgorithm 
    : public CommonAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    MagnetAlgorithm();

    /** Destructor */
    ~MagnetAlgorithm();

    /** Passes the element entry  */
    void passEntry(const MagnetData& mdata, Coordinates& p); 

    /** Passes the element exit  */
    void passExit(const MagnetData& mdata, Coordinates& p); 

   /** Applies the multipole kick */
   void applyMltKick(const MagnetData& mdata, double rkicks, Coordinates& p);

  protected:
   
    /** Applies the multipole kick */
    void applyMltKick(PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p);

  };

}

#include "TEAPOT/Integrator/MagnetAlgorithm.icc"

#endif
