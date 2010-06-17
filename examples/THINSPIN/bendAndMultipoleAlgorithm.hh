// Library       : TEAPOT
// File          : TEAPOT/Integrator/bendAndMultipoleAlgorithm.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_THINSPIN_BEND_AND_MULTIPOLE_ALGORITHM_HH
#define UAL_THINSPIN_BEND_AND_MULTIPOLE_ALGORITHM_HH

#include "magnetAlgorithm.hh"
#include "getB1tw.cc"
#include "getB2tw.cc"

namespace THINSPIN {
 
  /** A template of the dipole algorithm used by the conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class bendAndMultipoleAlgorithm 
    : public magnetAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    bendAndMultipoleAlgorithm();

    /** Destructor */
    ~bendAndMultipoleAlgorithm();

    /** Propagates a probe coordinates through the bend*/
//  void passBend(const TEAPOT::DipoleData& ddata, const TEAPOT::MagnetData& mdata, Coordinates& p, Coordinates& tmp, double v0byc);
    void passBend(const TEAPOT::DipoleData& ddata, const TEAPOT::MagnetData& mdata, Coordinates& p, Coordinates& tmp, double v0byc, int ip);

    /** Propagates a probe coordinates through the bend slice*/
//  void passBendSlice(const TEAPOT::ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);
    void passBendSlice(const TEAPOT::ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc, int ip);

    /** Applies a thin bend kick*/
    void applyThinBendKick(const TEAPOT::DipoleData& data, 
                           const TEAPOT::MagnetData& mdata, double rkicks,
                           Coordinates& p, double v0byc, int ip);

    /** Calculates the delta path*/
    void deltaPath(const TEAPOT::ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

  };

}

#include "bendAndMultipoleAlgorithm.icc"

#endif
