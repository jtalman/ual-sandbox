std::cout     << "#################################   Design Beam Orientation\n";
double gamma0  = 2.278;                            // fundamental kinematic parameter
double c       = 1.;                               // other units (mks) have 2.99792458e8 m/s
double b0      = sqrt(1.-1./gamma0/gamma0);        // equivalent fundamental kinematic parameter
double v0      = b0*c;                             // equivalent fundamental kinematic parameter

// $UAL/codes/PAC/src/PAC/Beam/BeamAttributes.hh   // # (index) of member variable
double m0      = UAL::pmass;                       // 2
double e0      = gamma0*m0;                        // 1
double p0      = gamma0*m0*v0;                     //

// double q0      = UAL::elemCharge;               // 3
double q0      = 1.0;                              // 3
double t0      = 0.;                               // 4
// double f0      = 1;                             // 5
// double f0      = 646617.830;//678910;//541426.7816;                      // 5
double M0      = 1.;                               // 6
double G0      = UAL::pG;                          // 7
double g0      = UAL::pg;                          // 7
std::cout     << "#################################   Design Beam Orientation\n";
