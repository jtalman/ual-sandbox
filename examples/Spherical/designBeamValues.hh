#define GAMMA_FROZEN_SPIN 1.248107349

std::cout     << "#################################   Design Beam Orientation\n";
double gamma0  = GAMMA_FROZEN_SPIN;                // fundamental kinematic parameter
double c       = 1;                                // other units (mks) have 2.99792458e8 m/s
double b0      = sqrt(1-1/gamma0/gamma0);          // equivalent fundamental kinematic parameter

// $UAL/codes/PAC/src/PAC/Beam/BeamAttributes.hh   // # (index) of member variable
double m0      = UAL::pmass;                       // 2
double e0      = gamma0*m0;                        // 1
double p0      = gamma0*m0*b0*c;

double q0      = UAL::elemCharge;                  // 3
double t0      = 0;                                // 4
double f0      = 1;                                // 5
double M0      = 1;                                // 6
double G0      = UAL::pG;                          // 7
double R0      = 30;                               // 10
double L0      = R0*p0;                            // 8
double E0      = 10.5e6;                           // 9

double gap     = 3e-2;                             // should be 11?

std::cout     << "#################################   Design Beam Orientation\n";

double k       = R0*p0*b0;                         //

//                   probe deviations
double  dx     = 0.001;                            // main input
double  dy     = 0.0001; 
double  dz     = 0.0; 
double dpx     = 0.0; 
double dpy     = 0.0; 
//                   probe deviations

//                   Case I: dx and implied dE
double Rin     = R0+dx;                            //
double gamma   = gamma0;                           //
double vin     = sqrt(1-1/gamma/gamma);
double Ein     = gamma*m0*c*c-k/Rin;               // E for compatibility with Munoz
//                   Case I: dx and implied dE

double Edes    = m0*c*c/gamma0;                    // Design Energy (Munoz potential)
double dE      = Ein-Edes; 

double p5Input = dE/p0;
