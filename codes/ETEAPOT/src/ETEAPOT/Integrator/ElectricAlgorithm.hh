// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/ElectricAlgorithm.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_ETEAPOT_ELECTRIC_ALGORITHM_HH
#define UAL_ETEAPOT_ELECTRIC_ALGORITHM_HH

#include "SMF/PacElemMultipole.h"
#include "ETEAPOT/Integrator/ElectricData.hh"
#include "ETEAPOT/Integrator/CommonAlgorithm.hh"

namespace ETEAPOT {
 
  /** A template of the common methods used by the electric conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class ElectricAlgorithm
    : public CommonAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    ElectricAlgorithm();

    /** Destructor */
    ~ElectricAlgorithm();

    /** Passes the element entry  */
    void passEntry(const ElectricData& edata, Coordinates& p);

    /** Passes the element exit  */
    void passExit(const ElectricData& edata, Coordinates& p);

   /** Applies the multipole kick */
   void applyMltKick(const ElectricData& edata, double rkicks, Coordinates& p);

  protected:
   
    /** Applies the multipole kick */
    void applyMltKick(PacElemMultipole* mult, PacElemOffset* offset, double rkicks, Coordinates& p);

  };

}

#include "ETEAPOT/Integrator/ElectricAlgorithm.icc"

#endif
