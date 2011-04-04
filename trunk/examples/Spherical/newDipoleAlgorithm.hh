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

    void traverseSplitBendExactly(const ETEAPOT::ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc, const PAC::BeamAttributes cba,double R0, double splitTheta);
    void handleSplitBendBoundary(Coordinates& p,const PAC::BeamAttributes cba,double R0);

    double getPotentialEnergy(double Eel0,double R0,double r){
std::cout << "JDT - enter double getPotentialEnergy(double Eel0,double R0,double r)\n";
std::cout << "k          " << k          << "\n";
std::cout << "Eel0*R0*R0 " << Eel0*R0*R0 << "\n";
       double value = -k*(1/r-1/R0);
       return value;
//     return -Eel0*R0*(R0/r-1);                   //  (2) page 15, "...Code...", Feb 17, 2011
// E field    -E0*R0*R0/r/r
//     return E0*R0*log(r/R0); 
    }

    double PE(double Rsxf,double r){
        double value = -k*(1/r-1/Rsxf);
        return value;
    }

    double Cxi(double Q,double theta){             // (30) page 22, "...Code...", Feb 17, 2011
        double value=cos(Q*theta);
        return value;
    }

    double Cxip(double Q,double theta){            // (30) page 22, "...Code...", Feb 17, 2011
        double value=-sin(Q*theta)*Q;
        return value;
    }

    double Sxi(double Q,double theta){             // (30) page 22, "...Code...", Feb 17, 2011
        double value=sin(Q*theta)/Q;
        return value;
    }

    double Sxip(double Q,double theta){            // (30) page 22, "...Code...", Feb 17, 2011
        double value=cos(Q*theta);
        return value;
    }

    double get_pz(double g,double m0,double Eel0,double R0,double r,double px,double py){
        double  e  =g*m0;
        double me  =e-PE(r);
//      double me  =e-getPotentialEnergy(Eel0,R0,r);
        double pzSQ=me*me-px*px-py*py-m0*m0;
        return sqrt(pzSQ);
    }

    double get_uLc(double dl,double dr,double pl,double pr){
        return dl*pr-dr*pl;
    }

    double c;
    double gamma;
    double mass;
    double k;
    double E;
    double L;
    double lambda;
    double epsilon;
    double kappa;

    double h0;
    double h0p;
    double C;
    double theta0;

    double _h0(double r0){
        double value = L/mass/r0-k*gamma/L;
        return value;
    }

    double ht(double r){                       // h theta
        double fac   = k/L/mass/c/c;
        double value = L/mass/r-fac*(E+k/r);
        return value;
    }

    double htp(const Coordinates p,double Rsxf,double r){                      // h theta prime
        double drdtheta = Rsxf*p[1];
        double value    = -(L/mass/r/r)*drdtheta+(k*k/L/mass/c/c/r/r)*drdtheta;
        return value;
    }

    double htp2(const Coordinates p,double Rsxf,double r){                     // h theta prime
        double drdtheta = Rsxf*p[1];
        double value    = -kappa*kappa*(L/mass/r/r)*drdtheta;
        return value;
    }

    double hr(double theta){
        double value = C*sin( kappa*(theta-theta0) )/kappa;
        return value;
    }

    double _ht(double theta){
        double value = C*cos( kappa*(theta-theta0) );
        return value;
    }

    double _theta0(){
//  double _theta0(double h0,double h0p){
//  double theta0(const Coordinates p,double Rsxf,double r){
        return atan( h0p/h0/kappa )/kappa;
    }

    double CSQ(){
//  double CSQ(double h0,double h0p){
        return h0*h0+h0p*h0p/kappa/kappa;
    }

    double get_rFromProbe(double x,double y,double z){
       return sqrt(x*x+y*y+z*z);
    }

    double get_rFromEllipse(double theta){
        double fac = L*mass*c*c/k/E;
        return lambda/( 1+fac*C*cos( kappa*(theta-theta0) ) );
//      return lambda/(1+epsilon*cos(kappa*theta));
    }

#include "getTimeAlternate.inline"

    double get_timeFromFirstTermViaMaple(double fac,double theta){
        double a  = epsilon;
        double t  = tan(kappa*theta/2);
        double am = a-1;
        double ap = a+1;
        double pm = sqrt((1+a)*(1-a));
        double A  = 2*atan(am*t/pm)/kappa/am/pm;
        double B  = -2*a*t/kappa/am/ap/(-t*t-1+a*t*t-a);
        double C  = -2*a*atan(am*t/pm)/kappa/am/ap/pm;
        return fac*(A+B+C);
    }

    double get_timeFromSecondTermViaMaple(double fac,double theta){
        double a  = epsilon;
        double t  = tan(kappa*theta/2);
        double ap = a+1;
        double am = a-1;
        double pm = sqrt((1+a)*(1-a));
        double D  = -2*atan(am*t/pm)/kappa/am/pm;
        double E  = 2*t/kappa/am/ap/(-t*t-1+a*t*t-a);
        double F  = 2*atan(am*t/pm)/kappa/am/ap/pm;
        return fac*(D+E+F);
    }

// http://integrals.wolfram.com/index.jsp?expr=1%2F%281%2Ba*cos%28k*x%29%29^2&random=false
    double get_timeFromFirstTermViaMathematica(double fac,double theta){          // can't be right
        double a  = epsilon;
        double am = sqrt(a*a-1);
        double t  = tan(kappa*theta/2);
        double G  = a*sin(kappa*theta)/am/am/kappa/( a*cos(kappa*theta)+1 );
        double H  = -2*a*atanh( (a-1)*t/am )/am/am/am/kappa;
        return fac*(G+H);
    }

// http://integrals.wolfram.com/index.jsp?expr=cos%28k*x%29%2F%281%2Ba*cos%28k*x%29%29^2&random=false
    double get_timeFromSecondTermViaMathematica(double fac,double theta){         // can't be right
        double a  = epsilon;
        double am = sqrt(a*a-1);
        double t  = tan(kappa*theta/2);
        double J  = 2*a*atanh( (a-1)*t/am )/am/am/am/kappa;
        double K  = -sin(kappa*theta)/am/am/kappa/( a*cos(kappa*theta)+1 );
        return fac*(J+K);
    }

  };

#include "newDipoleAlgorithm.icc"
#endif
