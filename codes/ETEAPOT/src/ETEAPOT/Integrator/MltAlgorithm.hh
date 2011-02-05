// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/MltAlgorithm.hh
// Copyright     : see Copyright file


#ifndef ETEAPOT_MLT_ALGORITHM_HH
#define ETEAPOT_MLT_ALGORITHM_HH

#include "SMF/PacElemMultipole.h"
#include "ETEAPOT/Integrator/MltData.hh"
#include "ETEAPOT/Integrator/CommonAlgorithm.hh"

namespace ETEAPOT {
 
  /** A template of the common methods used by the mlt conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class MltAlgorithm
    : public CommonAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    MltAlgorithm();

    /** Destructor */
    ~MltAlgorithm();

    /** Passes the element entry  */
    void passEntry(const MltData& edata, Coordinates& p);

    /** Passes the element exit  */
    void passExit(const MltData& edata, Coordinates& p);

   /** Applies the multipole kick */
   void applyMltKick(const MltData& edata, double rkicks, Coordinates& p);

  protected:
   
    /** Applies the multipole kick */
    void applyMltKick(PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p);

  };

}

#include "ETEAPOT/Integrator/MltAlgorithm.icc"

#endif
