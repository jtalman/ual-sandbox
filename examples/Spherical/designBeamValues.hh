#define GAMMA_FROZEN_SPIN 1.248107349

std::cout     << "#################################   Design Beam Orientation\n";
double gamma0  = GAMMA_FROZEN_SPIN;                // fundamental kinematic parameter
double c       = 1;                                // other units (mks) have 2.99792458e8 m/s
double b0      = sqrt(1-1/gamma0/gamma0);          // equivalent fundamental kinematic parameter
double v0      = b0*c;                             // equivalent fundamental kinematic parameter

// $UAL/codes/PAC/src/PAC/Beam/BeamAttributes.hh   // # (index) of member variable
double m0      = UAL::pmass;                       // 2
double e0      = gamma0*m0;                        // 1
double p0      = gamma0*m0*v0;                     //

double q0      = UAL::elemCharge;                  // 3
double t0      = 0;                                // 4
double f0      = 1;                                // 5
double M0      = 1;                                // 6
double G0      = UAL::pG;                          // 7
double R0      = 30;                               // 10
double L0      = R0*p0;                            // 8
double El0     = 10.5e6;                           // 9

double gap     = 3e-2;                             // should be 11?

std::cout     << "#################################   Design Beam Orientation\n";
