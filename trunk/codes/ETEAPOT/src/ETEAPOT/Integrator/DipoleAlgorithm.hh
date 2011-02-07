// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/DipoleAlgorithm.hh
// Copyright     : see Copyright file


#ifndef ETEAPOT_DIPOLE_ALGORITHM_HH
#define ETEAPOT_DIPOLE_ALGORITHM_HH

#include "ETEAPOT/Integrator/MltAlgorithm.hh"

namespace ETEAPOT {

  /** A template of the dipole algorithm used by the conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class DipoleAlgorithm
    : public MltAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    DipoleAlgorithm();

    /** Destructor */
    ~DipoleAlgorithm();

    /** Propagates a probe coordinates through the bend*/
    void passBend(const DipoleData& ddata, const MltData& mdata,
                  Coordinates& p, Coordinates& tmp,
                  const PAC::BeamAttributes& ba);

    /** Propagates a probe coordinates through the bend slice*/
    void passBendSlice(const ElemSlice& slice, 
                       Coordinates& p, Coordinates& tmp,
                       double v0byc);

    /** Applies a thin bend kick*/
    void applyThinBendKick(const DipoleData& data,
                           const MltData& mdata, double rkicks,
                           Coordinates& p,
                           double v0byc);

    /** Calculates the delta path*/
    void deltaPath(const ElemSlice& slice, 
                   Coordinates& p, Coordinates& tmp,
                   double v0byc);

  public:

      void addPotential(double r0, double m, Coordinates& p, double v0byc);
      void removePotential(double r0, double m, Coordinates& p, double v0byc);
  };

}

#include "ETEAPOT/Integrator/DipoleAlgorithm.icc"

#endif
