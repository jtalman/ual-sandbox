#ifndef UAL_CYLINDRICAL_DIPOLE_TRACKER_HH
#define UAL_CYLINDRICAL_DIPOLE_TRACKER_HH

#include <cstdlib>
#include <cmath>

#include <iomanip>
#include "UAL/Common/Def.hh"
#include "ETEAPOT/Integrator/MltAlgorithm.hh"

  /** A template of the dipole algorithm used by the conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class newDipoleAlgorithm 
    : public ETEAPOT::MltAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    newDipoleAlgorithm();

    /** Destructor */
    ~newDipoleAlgorithm();

    /** Propagates a probe coordinates through the bend*/
    void passBend(const ETEAPOT::DipoleData& ddata, const ETEAPOT::MltData& mdata, Coordinates& p, Coordinates& tmp, double v0byc, const PAC::BeamAttributes cba);

    /** Propagates a probe coordinates through the bend slice*/
    void passBendSlice(const ETEAPOT::ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

    /** Applies a thin bend kick*/
    void applyThinBendKick(const ETEAPOT::DipoleData& data, 
                           const ETEAPOT::MltData& mdata, double rkicks,
                           Coordinates& p, double v0byc);

    /** Calculates the delta path*/
    void deltaPath(const ETEAPOT::ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

    void enterBendCorrection(Coordinates& p,const PAC::BeamAttributes cba);
    void traverseSplitBendExactly(const ETEAPOT::ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc, const PAC::BeamAttributes cba,double R0, double splitTheta);
    void handleSplitBendBoundary(Coordinates& p,const PAC::BeamAttributes cba);
    void leaveBendCorrection(Coordinates& p,const PAC::BeamAttributes cba);

    double getPotentialEnergy(double q0,double E0,double R0,double r){
       return q0*E0*R0*log(r/R0); 
    }

    double Cxi(double Q,double theta,double xi0){
        double value=cos(Q*theta)*(1-xi0)+xi0;
        return value;
    }

    double CxiP(double Q,double theta,double xi0){
        double value=-sin(Q*theta)*Q*(1-xi0);
        return value;
    }

    double Sxi(double Q,double theta,double xi0){
        double value=sin(Q*theta)/Q-cos(Q*theta)*xi0+xi0;
        return value;
    }

    double SxiP(double Q,double theta,double xi0){
        double value=cos(Q*theta)+sin(Q*theta)*Q*xi0;
        return value;
    }

  };

#include "newDipoleAlgorithm.icc"
#endif
