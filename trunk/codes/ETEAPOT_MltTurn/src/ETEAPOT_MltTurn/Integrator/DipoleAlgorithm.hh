#ifndef UAL_M_1_MP_CV_MLT_TURN_HH
#define UAL_M_1_MP_CV_MLT_TURN_HH

#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
//#include <stdlib.h>

//#include "SMF/PacElemMultipole.h"

#include "UAL/Common/Def.hh"
#include "ETEAPOT_MltTurn/Integrator/MltAlgorithm.hh"
#include "ETEAPOT_MltTurn/Integrator/Matrices.hh"

  /** A template of the dipole algorithm used by the conventional tracker and DA integrator */

  template<class Coordinate, class Coordinates> class DipoleAlgorithm 
    : public ETEAPOT_MltTurn::MltAlgorithm<Coordinate, Coordinates> {

  public:

    /** Constructor */
    DipoleAlgorithm();

    /** Destructor */
    ~DipoleAlgorithm();

//                   STATIC
    static int turn;

    static std::string bend_m_elementName[8192];
    static double bend_m_sX[8192];

    static double spin[41][3];
    static double dZFF;

//                   GLOBALS
    double c;          // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/algorithm.icc +73
    double gamma;      // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/reference.inline
    double mass;       // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/reference.inline
    double k;          // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/algorithm.icc +78
    double EscM;       // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/reference.inline
    double L;          // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/reference.inline
    double lambda;     // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/hamilton.inline +4
    double epsilon;    // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/hamilton.inline +24
    double kappa;      // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/hamilton.inline +10

    double h0_tilda;   // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/hamilton.inline +19
    double h0p_tilda;  // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/hamilton.inline +28
    double C_tilda;    // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/hamilton.inline +30
    double theta0;     // $UAL/codes/ETEAPOT/src/ETEAPOT/inverseSquareBend/MunozPavic/conservedVector/hamilton.inline +29
//                   GLOBALS

    double getEpsilonTheta0(){
       double epsilonTheta0 = 0;
       double fac = sqrt(h0_tilda*h0_tilda+h0p_tilda*h0p_tilda/kappa/kappa);
              epsilonTheta0 = C_tilda*mass*c*c*L/EscM/k;
//            epsilonTheta0 =     fac*mass*c*c*L/EscM/k;
       return epsilonTheta0;
    }

    double getEpsilonMassagedMunoz35(const PAC::BeamAttributes cba,double Rsxf,double rIn){
       #include "ETEAPOT/Integrator/getDesignBeam.h"
       double g0 = e0/m0;
//     double epsilonMassagedMunoz35 = 0;
       double EscM0 = m0*c*c/g0;
       double kappa0 = 1/g0;
       double dEscMbyEscM = (EscM-EscM0)/EscM;
       double dEscM = EscM-EscM0;
       double dK = kappa-kappa0;
       double EscMS  = EscM+k/Rsxf;
       double h0_tildaA = L/m0/rIn-k*EscMS/L/m0/c/c;
//            h0_tilda  = ht_tilda(rIn);
       double Lcbyk = L*c/k;
       double epsilonMassagedMunoz35 = Lcbyk*sqrt( 2*EscM0*dEscM+dEscM*dEscM-(m0*c*c)*(m0*c*c)*(2*kappa0*dK+dK*dK) );
              epsilonMassagedMunoz35 = epsilonMassagedMunoz35/EscM;
       return epsilonMassagedMunoz35;
    }

    void getE(int ip,const ETEAPOT::DipoleData& ddata, const ETEAPOT::MltData& mdata, Coordinates& p, Coordinates& tmp, double v0byc, const PAC::BeamAttributes cba);

    /** Passes the element entry  */
    void passEntry(int ip, const ETEAPOT::MltData& edata, Coordinates& p, int mltK, double m_m, const PAC::BeamAttributes cba){
//   std::cerr << "passEntry for bend stub\n";
    }

    /** Passes the element exit  */
    void passExit(int ip, const ETEAPOT::MltData& edata, Coordinates& p, int mltK, double m_m, const PAC::BeamAttributes cba){
//   std::cerr << "passExit  for bend stub\n";
    }

    /** Propagates a probe coordinates through the bend*/
    void passBend(int ip,const ETEAPOT::DipoleData& ddata, const ETEAPOT::MltData& mdata, Coordinates& p, Coordinates& tmp, double v0byc, const PAC::BeamAttributes cba, int bend );

    /** Propagates a probe coordinates through the bend slice*/
    void passBendSlice(const ETEAPOT::ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

    /** Applies a thin bend kick*/
    void applyThinBendKick(const ETEAPOT::DipoleData& data, 
                           const ETEAPOT::MltData& mdata, double rkicks,
                           Coordinates& p, double v0byc);

    /** Calculates the delta path*/
    void deltaPath(const ETEAPOT::ElemSlice& slice, Coordinates& p, Coordinates& tmp, double v0byc);

    void traverseSplitBendExactly(int ip, Coordinates& p, Coordinates& tmp, double v0byc, const PAC::BeamAttributes cba,double Rsxf, double splitTheta);
    void handleSplitBendBoundary(Coordinates& p,const PAC::BeamAttributes cba,double Rsxf);
    void splitBendKick(Coordinates& p,const PAC::BeamAttributes cba,double Rsxf,double m,double theta);

    double getPotentialEnergy(double Eel0,double Rsxf,double r){
       double value = -k*(1/r-1/Rsxf);
       return value;
    }

    double PE(double Rsxf,double r){
// Appendix UALcode: Development of the UAL/ETEAPOT Code for the Proton EDM Experiment 8/28/2012
// Eq (2) Page 13 with probe charge, q0, included
// V(r)=-q0*E0*r0( (r0/r)^m - 1)/m
//     =-q0*E0*r0( (r0/r) - 1)
//     =-q0*E0*r0^2(1/r-1/r0)
//     =-k(1/r-1/r0)
        double value = -k*(1/r-1/Rsxf);
        return value;
    }

    double get_pz(double g,double m0,double Eel0,double Rsxf,double r,double px,double py){
        double  e  =g*m0;
        double me  =e-PE(r);
        double pzSQ=me*me-px*px-py*py-m0*m0;
        return sqrt(pzSQ);
    }

    double get_uLc(double dl,double dr,double pl,double pr){
        return dl*pr-dr*pl;
    }

    double _h0_tilda(double rIn){
        double value = L/mass/rIn-k*gamma/L;
        return value;
    }

    double ht_tilda(double r){                       // h theta
        double fac   = k/L/mass/c/c;
        double value = L/mass/r-fac*(EscM+k/r);
        return value;
    }

    double htp_tilda(const Coordinates p,double Rsxf,double r){                      // h theta prime
        double drdtheta = Rsxf*p[1];
        double value    = -(L/mass/r/r)*drdtheta+(k*k/L/mass/c/c/r/r)*drdtheta;
        return value;
    }

    double htp_tilda2(const Coordinates p,double Rsxf,double r){                     // h theta prime
        double drdtheta = Rsxf*p[1];
        double value    = -kappa*kappa*(L/mass/r/r)*drdtheta;
        return value;
    }

    double htp_tilda3(const Coordinates p,double Rsxf,double r,double vz){           // h theta prime
        double value = -kappa*kappa*gamma*p[1]*vz;
        return value;
    }

    double htp_tilda4(const Coordinates p,double Rsxf,double r){                    // h theta prime
        double value = -kappa*kappa*(L/mass/r)*p[1];
        return value;
    }

    double hr_tilda(double theta){
        double value = C_tilda*sin( kappa*(theta-theta0) )/kappa;
        return value;
    }

    double _ht_tilda(double theta){
        double value = C_tilda*cos( kappa*(theta-theta0) );
        return value;
    }

    double _theta0(){
        double value = atan2( h0p_tilda/kappa,h0_tilda )/kappa;
//      double value = atan( h0p_tilda/h0_tilda/kappa )/kappa;
//      if(value<-UAL::pi/2){value=UAL::pi-value;}
//      if(value<-UAL::pi/2){value=-value;}
//      if(value<-UAL::pi/2){value=value+UAL::pi;}
        return value;
    }

    double C_tildaSQ(){
        return h0_tilda*h0_tilda+h0p_tilda*h0p_tilda/kappa/kappa;
    }

    double get_rFromProbe(double x,double y,double z){
       return sqrt(x*x+y*y+z*z);
    }

    double get_rFromEllipse(double theta){
        double fac = L*mass*c*c/k/EscM;
        return lambda/( 1+fac*C_tilda*cos( kappa*(theta-theta0) ) );
    }

#include "ETEAPOT/Integrator/getTimeAlternate.inline"
#include "ETEAPOT/Integrator/timeViaHyperbolic"
#include "ETEAPOT/Integrator/timeViaExpansion"

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

#include "DipoleAlgorithm.icc"
#endif
