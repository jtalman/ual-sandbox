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
  std::cout << "usage: ./xyPlotLambda typicalRadius LcOver_k  Efac  grid  oscillations\n";
  std::cout << "                        (lambda)                               \n";
  std::cout << "e.g. (single ellipse - first, leftmost, ellipse in Munozs Figure 3)\n";
  std::cout << "       ./xyPlotLambda       2        3.23     0.99   100       1\n";
  std::cout << "or  (reproduce Munozs Figure 3 - 16 ellipses)\n";
  std::cout << "       ./xyPlotLambda 2 3.23 0.99 400 16\n";
  exit(1);
 }

 double lambda  = atof(argv[1]);
 std::cout << "lambda  "    << lambda << "\n";

 double LcOver_k= atof(argv[2]);
 std::cout << "LcOver_k"    << LcOver_k<< "\n";
 double kapSqu  = 1-1/LcOver_k/LcOver_k;
 std::cout << "kapSqu  "    << kapSqu << "\n";
 double kap     = sqrt(kapSqu);
 std::cout << "kap     "    << kap    << "\n";
 double Efac    = atof(argv[3]);
 std::cout << "Efac    "    << Efac   << "\n";
 double eps     = LcOver_k*sqrt(1-kapSqu/Efac/Efac);
 std::cout << "eps     "    << eps    << "\n";
 std::cout <<                            "\n";

 std::ofstream fp("xyPlotLambda.out");
 double Ngrid   = atof(argv[4]);
 double delThta = 2*M_PI/Ngrid;
 int oscills    = atof(argv[5]);
 double     tau = oscills*2*M_PI;
 double    Thta = 0;
 double    r    = 0;
 double    x    = 0;
 double    y    = 0;
 for(Thta=0;Thta<tau;Thta+=delThta){
  r = lambda/(1+eps*cos(kap*Thta));
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

 double kOverE  = lambda/(LcOver_k*LcOver_k-1);
 std::cout << "k/E     "    << kOverE << " (not strictly needed)\n";
 double k       = kOverE*Efac*mpcSqu;
 std::cout << "k       "    << k      << " (not strictly needed)\n";

 return 0;
}
