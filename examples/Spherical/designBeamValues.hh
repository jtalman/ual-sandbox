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

#define GAMMA_FROZEN_SPIN 1.248107349

//                   constants
  double m_p = UAL::pmass;                    // GeV  
//double m_p = 0.93827231;                    // GeV  
  double c = 1;                               // 2.99792458e8 m/s
  double gamma_0 = GAMMA_FROZEN_SPIN;         //      
  double beta_0 = sqrt(1-1/gamma_0/gamma_0);  //
  double p_0c = gamma_0*m_p*beta_0;           // GeV  
//double p_0c = 0.7007405278;                 // GeV  
  double R0_= 40;                             //      
  double k = R0_*p_0c*beta_0;                 //      
//double k = 16.77233867;                     //      
//                   constants

//                   probe deviations
double  dx  =  0.01;             // main input
double  dy  =  0.0; 
double  dz  =  0.0; 
double dpx  =  0.0; 
double dpy  =  0.0; 
double dpz;                      // main ouput
//                   probe deviations

//                   Case I: dx and implied dE
  double r = R0_+dx;                          //      
  double gamma = gamma_0;                     //      
  double E = gamma*m_p*c*c-k/r;               //      
  double L = r*p_0c;                          //      
//                   Case I: dx and implied dE

//                   probe values
double x  = R0_+dx; 
double y  = dy;
double z  = dz;

double px = dpx;
double py = dpy;
double pz;

double vin;
double pin;

vin = sqrt(1-1/gamma/gamma);
pin = gamma*m_p*vin;
std::cout << "vin " << vin  << "\n";
std::cout << "pin " << pin << "\n";

pz = sqrt(pin*pin-px*px-py*py);
std::cout << "pz " << pz << "\n";

double Lx = y*pz-z*py;
double Ly = z*px-x*pz;
double Lz = x*py-y*px;
double L_ = sqrt(Lx*Lx+Ly*Ly+Lz*Lz);

double E0 = m_p*c*c/gamma_0;
double kap0 = 1/gamma_0;
double dEbyE = (E-E0)/E;
double dE = E-E0; 

double p5Input=dE/p_0c;

std::cout << "#################################   Design Beam Values\n";
