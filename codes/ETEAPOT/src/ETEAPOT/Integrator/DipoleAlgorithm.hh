// Library       : TEAPOT
// File          : TEAPOT/Integrator/DipoleAlgorithm.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_TEAPOT_DIPOLE_ALGORITHM_HH
#define UAL_TEAPOT_DIPOLE_ALGORITHM_HH

#include "TEAPOT/Integrator/MagnetAlgorithm.hh"

namespace TEAPOT {
 
  /** A template of the dipole algorithm used by the conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class DipoleAlgorithm 
    : public MagnetAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    DipoleAlgorithm();

    /** Destructor */
    ~DipoleAlgorithm();

    /** Propagates a probe coordinates through the bend*/
    void passBend(const DipoleData& ddata, const MagnetData& mdata, Coordinates& p, Coordinates& tmp, double v0byc);

    /** Propagates a probe coordinates through the bend slice*/
    void passBendSlice(const ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

    /** Applies a thin bend kick*/
    void applyThinBendKick(const DipoleData& data, 
                           const MagnetData& mdata, double rkicks,
                           Coordinates& p, double v0byc);

    /** Calculates the delta path*/
    void deltaPath(const ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

  };

}

#include "TEAPOT/Integrator/DipoleAlgorithm.icc"

#endif
