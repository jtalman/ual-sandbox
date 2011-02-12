std::cout << "#################################   Design Beam Values\n";
double gamma0=1.248107349;
std::cout << "   double gamma0= " << gamma0 << "\n";
std::cout << "\n";
double m0=UAL::pmass;                       // * (2)

double e0=gamma0*m0;                        // 1
//double m0=;                               // 2
double q0=UAL::elemCharge;                  // 3
double t0=0;                                // 4
//double f0=0;                                // 5
double M0=1;                                // 6
double G0=UAL::pG;                          // 7
//double R0=40;                               // * (10)
double v0=sqrt(1-1/gamma0/gamma0);
double p0=gamma0*m0*v0;
//double L0=40.*p0;                            // 8
//double E0=12e6;                             // 9
//double R0=;                               // 10
//double gap=3e-2;                          // 11?
std::cout << "   double e0= " << e0 << "\n";
std::cout << "   double m0= " << m0 << "\n";
std::cout << "   double q0= " << q0 << "\n";
std::cout << "   double t0= " << t0 << "\n";
//std::cout << "   double f0= " << f0 << "\n";
std::cout << "   double M0= " << M0 << "\n";
std::cout << "   double G0= " << G0 << "\n";

//std::cout << "   double L0= " << L0 << "\n";
//std::cout << "   double E0= " << E0 << "\n";
//std::cout << "   double R0= " << R0 << "\n";
std::cout << "\n";
std::cout << "   double v0= " << v0 << "\n";
std::cout << "   double p0= " << p0 << "\n";
std::cout << "#################################   Design Beam Values\n";
