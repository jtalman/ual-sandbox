#include<iostream>
#include<fstream>
#include<cmath>
#include<stdlib.h>
#include<cstdlib>
#include <iomanip>
#include"../../../UAL/src/UAL/Common/Def.hh"

#define GAMMA_FROZEN_SPIN 1.248107349

int main(int argc,char* argv[]){
 std::cout << setiosflags( std::ios::showpos   );
 std::cout << setiosflags( std::ios::uppercase );
 std::cout << setiosflags( std::ios::scientific );
//std::cout << setw( 11 );
 std::cout << std::setfill( ' ' );
 std::cout << setiosflags( std::ios::left );
 std::cout << std::setprecision(13) ;

 std::cout << "JDT: file " << __FILE__ << " line " << __LINE__ << " argc " << argc << "\n";
 for(int i=0;i<argc;i++){
  std::cout << i << " " << argv[i] << "\n";
 }

 if(argc!=7){
  std::cout << "usage: ./orbit        E0            R0     Lfac        Efac    grid  oscillations\n";
  std::cout << "e.g.\n";
  std::cout << "       ./orbit +1.3976944506760E+07 30 1.671181439 .8012131335 100       2\n";
  exit(1);
 }

 double mp      = UAL::pmass;
 double q0      = UAL::elemCharge;
 double GeVperJ = 1./q0/1.e9;
 double c       = UAL::clight;
 double gamma0  = GAMMA_FROZEN_SPIN;
 double b0      = sqrt(1-1/gamma0/gamma0);
 double v0      = b0*c;

 std::cout << "mp [Gev] "   << mp     << "\n";
        mp      = mp/GeVperJ/c/c;
 std::cout << "mp [kg] "    << mp     << "\n";
 double mpc2    = mp*c*c;
 std::cout << "mp c^2  "    << mpc2   << "\n";
 double p0      = gamma0*mp*b0*c;
 double Egy0    = mpc2/gamma0;
 double kap0    = 1/gamma0;
 std::cout << "Egy0(Mun)  " << Egy0   << "\n";
 std::cout << "kap0       " << kap0   << "\n";

 std::cout << "c      "     << c      << "\n";
 std::cout << "gamma0 "     << gamma0 << "\n";
 std::cout << "b0     "     << b0     << "\n";
 std::cout << "p0     "     << p0     << "\n";
 std::cout << "v0     "     << v0     << "\n";
 std::cout << "q0     "     << q0     << "\n";

 double E0      = atof(argv[1]);  // design E field
 double r0      = atof(argv[2]);  // design radius
 double L0      = p0*r0;          // design angular momentum
 double k1      = q0*E0*r0*r0;
 double k2      = p0*v0*r0;
 double k3      = L0*v0;
 double k       = k1;
 std::cout << "E0(EField) "  << E0     << "\n";
 std::cout << "p0*v0/r0/q0"  << p0*v0/r0/q0<< "\n";

 std::cout << "r0(radius) "  << r0     << "\n";
 std::cout << "k1(forcfac) " << k      << "\n";
 std::cout << "k2(forcfac) " << k      << "\n";
 std::cout << "k3(forcfac) " << k      << "\n";
 std::cout << "k(forcfac) "  << k      << "\n";
 std::cout << "L0(angMom) "  << L0     << "\n";

 double Lfac    = atof(argv[3]);  // L factor
 double L       = Lfac*L0*b0;
 double EgyFac  = atof(argv[4]);  // (Munoz) energy factor
 double Egy     = EgyFac*mp*c*c;
 double deltEgy = Egy-Egy0;
 std::cout << "L(angMom)  " << L      << "\n";
 std::cout << "Egy(Mun)   " << Egy    << "\n";
 std::cout << "deltEgy    " << deltEgy<< "\n";

 std::cout << "RADIUS(k/E) " << k/Egy<< "\n";

 double kapSqu  = 1-k*k/L/L/c/c;
 std::cout << "kapSqu     " << kapSqu << "\n";
 double     kap = sqrt(kapSqu);
 std::cout << "kap        " << kap    << "\n";
 double deltkap = kap-kap0;
 std::cout << "deltkap    " << deltkap<< "\n";

 double dis     = 2*Egy0*deltEgy/mpc2/mpc2+deltEgy*deltEgy/mpc2/mpc2-2*kap0*deltkap-deltkap*deltkap;
// double dis     = 2*Egy0*deltEgy/mpc2/mpc2-2*kap0*deltkap;
 std::cout << "dis        " << dis    << "\n";

 double lambda1 = kapSqu*L*L*c*c/k/Egy;
 std::cout << "lambda1    " << lambda1<< "\n";
 double lambda2 = k/Egy;
        lambda2 = lambda2*(L*L*c*c/k/k-1);
 std::cout << "lambda2    " << lambda2<< "\n";
 double lambda  = lambda1;
 double C       = L*mp*c*c/k/Egy;
 std::cout << "C          " << C      << "\n";
 double h0      = c*sqrt(Egy*Egy/mp/mp/c/c/c/c-kapSqu);
 std::cout << "h0         " << h0     << "\n";
 double eps     = L*c/k;
 std::cout << "L*c/k      " << L*c/k  << "\n";
        eps     = eps*sqrt(1-mp*mp*c*c*c*c*kapSqu/Egy/Egy);
 std::cout << "eps        " << eps    << "\n";

 std::ofstream fp("orbit.out");
 double Ngrid   = atof(argv[5]);
 int oscills    = atof(argv[6]);
 double     tau = oscills*2*M_PI;
 double delThta = tau/Ngrid;
 double    Thta = 0;
 double    r    = 0;
 double    x    = 0;
 double    y    = 0;
 for(Thta=0;Thta<tau;Thta+=delThta){
  r = lambda/(1+C*h0*cos(kap*Thta));
  x = r * cos(Thta);
  y = r * sin(Thta);
         fp << x    << " " << y                             << "\n";
//       fp << Thta << " " << lambda/(1+C*h0*cos(kap*Thta)) << "\n";

//       fp << Thta << " " << cos(Thta) << "\n";
//std::cout << Thta << " " << cos(Thta) << "\n";
 }

 return 0;
}
