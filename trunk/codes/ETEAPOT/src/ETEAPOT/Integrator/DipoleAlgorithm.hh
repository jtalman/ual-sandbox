// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DipoleAlgorithm.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_ETEAPOT_DIPOLE_ALGORITHM_HH
#define UAL_ETEAPOT_DIPOLE_ALGORITHM_HH

#include "ETEAPOT/Integrator/ElectricAlgorithm.hh"

namespace ETEAPOT {
 
  /** A template of the dipole algorithm used by the conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class DipoleAlgorithm 
    : public ElectricAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    DipoleAlgorithm();

    /** Destructor */
    ~DipoleAlgorithm();

    /** Propagates a probe coordinates through the bend*/
    void passBend(const DipoleData& ddata, const ElectricData& mdata, Coordinates& p, Coordinates& tmp, double v0byc);

    /** Propagates a probe coordinates through the bend slice*/
    void passBendSlice(const ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

    /** Applies a thin bend kick*/
    void applyThinBendKick(const DipoleData& data, 
                           const ElectricData& mdata, double rkicks,
                           Coordinates& p, double v0byc);

    /** Calculates the delta path*/
    void deltaPath(const ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

  };

}

#include "ETEAPOT/Integrator/DipoleAlgorithm.icc"

#endif
