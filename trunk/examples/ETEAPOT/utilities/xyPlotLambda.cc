#include<iostream>
#include<fstream>
#include<cmath>
#include<stdlib.h>
#include<cstdlib>
#include <iomanip>
#include"../clr"
#include"../../../codes/UAL/src/UAL/Common/Def.hh"

#define GAMMA_FROZEN_SPIN 1.248107349

int main(int argc,char* argv[]){
 std::ofstream fp("xyPlotLambda.out");

#include "inputs"

 double kapSqu  = 1.-1./LcOver_k/LcOver_k;
 double kap     = sqrt(kapSqu);
 double eps     = LcOver_k*sqrt(1.-kapSqu/Efac/Efac);
 double R0=lambda/(1.+eps);

 double delThta = 2.*M_PI/Ngrid;
 double     tau = oscills*2.*M_PI;
 double    Thta = 0.;
 double    r    = 0.;
 double    x    = 0.;
 double    y    = 0.;
 for(Thta=0.;Thta<tau;Thta+=delThta){
  r = lambda/(1.+eps*cos(kap*Thta));
  x = r * cos(Thta);
  y = r * sin(Thta);
         fp << x    << " " << y                             << "\n";
 }

 double mp      = UAL::pmass;
 double q0      = UAL::elemCharge;
 double GeVperJ = 1./q0/1.e9;
 double c       = UAL::clight;
 
 std::cout << "mp [Gev] "   << mp     << " (not strictly needed)\n";
        mp      = mp/GeVperJ/c/c;
 std::cout << "mp [kg] "    << mp     << " (not strictly needed)\n";
 double mpcSqu  = mp*c*c;
 std::cout << "mp c^2  "    << mpcSqu << " (not strictly needed)\n";

 double kOverE  = lambda/(LcOver_k*LcOver_k-1.);
 std::cout << "k/E     "    << kOverE << " (not strictly needed)\n";
 double k       = kOverE*Efac*mpcSqu;
 std::cout << "k       "    << k      << " (not strictly needed)\n";
 double L=LcOver_k*k/c;
 std::cout << "L       "    << L << "\n";
 double E=k*1./kOverE;
 std::cout << "E       "    << E << "\n";

 double k2 = +2.68722e-09;
//double c = +2.99792e+08;
 double a = 1./LcOver_k;
 double kapSqu2 = 1.-a*a;
 std::cout << "k2      " << k2 << "\n";
 std::cout << "kapSqu2 " << kapSqu2 << "\n";

 double bSq=1./Efac/Efac-1.;
 std::cout << "bSq     " << bSq << "\n";

#include "outputs"

 return 0;
}