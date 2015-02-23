std::cout     << "#################################   Design Beam Orientation\n";
double g0      = 2.00231930420;//UAL::pg;          // 7
double gamma0  = sqrt(g0/(g0-2.));//UAL::eFSG;                      // fundamental kinematic parameter
double b0      = sqrt(1.-1./gamma0/gamma0);        // equivalent fundamental kinematic parameter

// $UAL/codes/PAC/src/PAC/Beam/BeamAttributes.hh   // # (index) of member variable
double m0      = UAL::emass;                       // 2
double e0      = gamma0*m0;                        // 1
double p0      = gamma0*m0*b0;                     //

// double q0      = UAL::elemCharge;               // 3
double q0      = 1.0;                              // 3
double t0      = 0.;                               // 4
// double f0      = 646617.830;//678910;//541426.7816;                      // 5
double M0      = 1.;                               // 6

std::cout     << "#################################   Design Beam Orientation\n";