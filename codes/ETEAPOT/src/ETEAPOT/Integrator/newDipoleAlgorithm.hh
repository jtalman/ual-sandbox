// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/newDipoleAlgorithm.hh
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#ifndef UAL_ETEAPOT_NEW_DIPOLE_ALGORITHM_HH
#define UAL_ETEAPOT_NEW_DIPOLE_ALGORITHM_HH

//#include <stdio.h>
//#include <stdlib.h>
  #include <cstdlib>
//#include <math.h>
  #include <cmath>

#include <iomanip>
#include "UAL/Common/Def.hh"
#include "ETEAPOT/Integrator/ElectricAlgorithm.hh"

namespace ETEAPOT {
 
  /** A template of the dipole algorithm used by the conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class newDipoleAlgorithm 
    : public ElectricAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    newDipoleAlgorithm();

    /** Destructor */
    ~newDipoleAlgorithm();

    /** Propagates a probe coordinates through the bend*/
    void passBend(const DipoleData& ddata, const ElectricData& mdata, Coordinates& p, Coordinates& tmp, double v0byc, const PAC::BeamAttributes cba);

    /** Propagates a probe coordinates through the bend slice*/
    void passBendSlice(const ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

    /** Applies a thin bend kick*/
    void applyThinBendKick(const DipoleData& data, 
                           const ElectricData& mdata, double rkicks,
                           Coordinates& p, double v0byc);

    /** Calculates the delta path*/
    void deltaPath(const ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

    void enterBendCorrection(Coordinates& p,const PAC::BeamAttributes cba);
    void traverseSplitBendExactly(const ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc, const PAC::BeamAttributes cba);
    void handleSplitBendBoundary(Coordinates& p,const PAC::BeamAttributes cba);
    void leaveBendCorrection(Coordinates& p,const PAC::BeamAttributes cba);

    double getPotentialEnergy(double q0,double E0,double R0,double r){
       return q0*E0*R0*log(r/R0); 
    }

  };

}

#include "ETEAPOT/Integrator/newDipoleAlgorithm.icc"

#endif
