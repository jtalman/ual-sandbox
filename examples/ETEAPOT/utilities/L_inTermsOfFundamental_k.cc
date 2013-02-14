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

 if(argc!=6){
  std::cout << "usage: ./L_inTermsOfFundamental_k    k     Lfac  Efac   grid  oscillations\n";
  std::cout << "e.g.\n";
  std::cout << "       ./L_inTermsOfFundamental_k +2.0E-09 3.2   .99     100       2\n";
  exit(1);
 }

 double mp      = UAL::pmass;
 double q0      = UAL::elemCharge;
 double GeVperJ = 1./q0/1.e9;
 double c       = UAL::clight;

 std::cout << "mp [Gev] "   << mp     << "\n";
        mp      = mp/GeVperJ/c/c;
 std::cout << "mp [kg] "    << mp     << "\n";
 double mpc2    = mp*c*c;
 std::cout << "mp c^2  "    << mpc2   << "\n";

 std::cout << "c      "     << c      << "\n";
 std::cout << "q0     "     << q0     << "\n";

 double k       = atof(argv[1]);  // k viewed as fundamental
 double Lfac    = atof(argv[2]);  // L in terms of k
 std::cout << "k(forcfac) "  << k      << "\n";

 double L       = Lfac*k/c;
 double EFac    = atof(argv[3]);  // (Munoz) energy factor
 double E       = EFac*mp*c*c;
 std::cout << "L(angMom)  " << L      << "\n";
 std::cout << "E(Mun)     " << E      << "\n";

 double kapSqu  = 1-k*k/L/L/c/c;
 std::cout << "kapSqu     " << kapSqu << "\n";
 double     kap = sqrt(kapSqu);
 std::cout << "kap        " << kap    << "\n";

 double lambda1 = kapSqu*L*L*c*c/k/E;
 std::cout << "lambda1    " << lambda1<< "\n";
 double lambda2 = k/E;
        lambda2 = lambda2*(L*L*c*c/k/k-1);
 std::cout << "lambda2    " << lambda2<< "\n";
 double lambda  = lambda1;
 double C       = L*mp*c*c/k/E;
 std::cout << "C          " << C      << "\n";
 double h0      = c*sqrt(E*E/mp/mp/c/c/c/c-kapSqu);
 std::cout << "h0         " << h0     << "\n";
 double eps     = L*c/k;
 std::cout << "L*c/k      " << L*c/k  << "\n";
        eps     = eps*sqrt(1-mp*mp*c*c*c*c*kapSqu/E/E);
 std::cout << "eps        " << eps    << "\n";

 std::ofstream fp("orbit.out");
 double Ngrid   = atof(argv[4]);
 int oscills    = atof(argv[5]);
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
