#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "Main/Eteapot.h"

#include "UAL/UI/Shell.hh"

#include "PAC/Beam/Particle.hh"
#include "PAC/Beam/Spin.hh"

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "Optics/PacTMap.h"
#include "Integrator/TeapotElemBend.h"

#include "positionPrinter.hh"
#include "xmgracePrint.hh"

#include "ETEAPOT/Integrator/DipoleTracker.hh"

using namespace UAL;

int main(int argc,char * argv[]){
 if(argc!=3){
  std::cout << "usage: ./determineTwiss ./data/E_BM_P1.0.sxf +1 (>&! OUTP1.0)\n";
  std::cout << "argv[0] is this executable         - ./determineTwiss     \n";
  std::cout << "argv[1] is the input sxf file      - ./data/E_BM_P1.0.sxf \n";
  std::cout << "argv[2] is the nominal electrode m - +1                   \n";
  exit(0);
 }

  ofstream m_m;
  m_m.open ("m_m");
  m_m << argv[2];
  m_m.close();

 std::string mysxf    =argv[1];
 std::string mysxfbase=mysxf.substr(7,mysxf.size()-11);
 std::cout << "mysxf     " << mysxf.c_str() << "\n";
 std::cout << "mysxfbase " << mysxfbase.c_str() << "\n";

 UAL::Shell shell;

 #include "designBeamValues.hh"
 #include "setBeamAttributes.hh"
 PAC::BeamAttributes& ba = shell.getBeamAttributes();

 #include "extractParameters.h"

 #include "simulatedProbeValues"

 // ************************************************************************
 std::cout << "\nDefine the space of Taylor maps." << std::endl;
 // ************************************************************************

  shell.setMapAttributes(UAL::Args() << UAL::Arg("order", order));

 // ************************************************************************
 std::cout << "\nBuild lattice." << std::endl;
 // ************************************************************************

 shell.readSXF(UAL::Args() << UAL::Arg("file",  sxfFile.c_str()));

 // ************************************************************************
 std::cout << "\nAdd split ." << std::endl;
 // ************************************************************************

  
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sbend")      << UAL::Arg("ir", splitForBends));
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Quadrupole") << UAL::Arg("ir", splitForQuads));
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sextupole")  << UAL::Arg("ir", splitForSexts));

 // ************************************************************************
 std::cout << "Select lattice." << std::endl;
 // ************************************************************************

 shell.use(UAL::Args() << UAL::Arg("lattice", "ring"));

 // ************************************************************************
 std::cout << "\nWrite SXF file ." << std::endl;
 // ************************************************************************

 shell.writeSXF(UAL::Args() << UAL::Arg("file",  outputFile.c_str()));

 // ************************************************************************
 std::cout << "\nDefine beam parameters." << std::endl;
 // ************************************************************************

 // ************************************************************************
 std::cout << "\nLinear analysis." << std::endl;
 // ************************************************************************
  
 // ************************************************************************
 std::cout << "\nAlgorithm Part. " << std::endl;
 // ************************************************************************

 UAL::APDF_Builder apBuilder;

 apBuilder.setBeamAttributes(ba);

 UAL::AcceleratorPropagator* ap = apBuilder.parse(apdfFile);

 if(ap == 0) {
   std::cout << "Accelerator Propagator has not been created " << std::endl;
   return 1;
 }

/*
 #define MAXPATHLEN 100
char cpath[MAXPATHLEN];
getcwd(cpath, MAXPATHLEN);
printf("pwd -> %s\n", cpath);
std::string path=cpath;
Eteapot* etpot;
double a0x=     (double)0;
double b0x=     (double)0;
double mu_xTent=(double)0;
double a0y=     (double)0;
double b0y=     (double)0;
double mu_yTent=(double)0;
//etpot->twissFromTracking( ba, ap, atof(argv[2]),a0x,b0x,mu_xTent,a0y,b0y,mu_yTent );
std::cerr << "RMT: a0x " << a0x << " b0x " << b0x << " mu_xTent " << mu_xTent << " a0y " << a0y << " b0y " << b0y << " mu_yTent " << mu_yTent << "\n";

 std::cout << "\n SXF_TRACKER tracker, ";
 std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;
*/

/*
 UAL::PropagatorSequence& apSeq = ap->getRootNode();

 int counter = 0;
 std::list<UAL::PropagatorNodePtr>::iterator it;
 for(it = apSeq.begin(); it != apSeq.end(); it++){
  std::cout << counter++ << " (*it)->getType() " << (*it)->getType() << std::endl;
 }
*/

//#include "trtrin"
//#include "trtrout"
double a0x=     (double)0;
double b0x=     (double)0;
double mu_xTent=(double)0;
double a0y=     (double)0;
double b0y=     (double)0;
double mu_yTent=(double)0;
//int turns=1;
      turns=1;
//#include "probeDataForTwiss"
ap -> propagate(bunch);

std::ofstream output;
char buffr2 [10];
sprintf( buffr2,"%+5.2f",atof(argv[2]) );
std::string bp2(buffr2);
std::string sT = "out/TWISS/TWISS_m=";
            sT+=bp2;

  output.open( sT.c_str() );
//output.open( filename.c_str() );
//output.open("TWISS");
#define PI 3.141592653589793
                                               // JDT (from RMT)
                                               // 7/13/2012
#include "trtrout"                             // Once around matrix determination begins at this point
double MX11=rx[1][1];double MX12=rx[1][2];
double MX21=rx[2][1];double MX22=rx[2][2];
double MXtr=MX11+MX22;
double cosMuX=MXtr/2;
if( (1-MXtr*MXtr/4)<0 ){std::cerr << "X: Trying to take square root of a negative number!\n";exit(1);}
double betaX=abs(MX12)/sqrt(1-MXtr*MXtr/4);
double sinMuX=MX12/betaX;
double alphaX=(MX11-MX22)/2/sinMuX;
output << "JDT: betaX  " << betaX  << "\n";
output << "JDT: cosMuX " << cosMuX << "\n";
output << "JDT: sinMuX " << sinMuX << "\n";
output << "JDT: alphaX " << alphaX << "\n";
double MuX_PR=acos(cosMuX);
double MuX;
                                               // half integer tune ambiguity resolution
                                               // NOT full integer tune ambiguity
if     (cosMuX>=0 && sinMuX>=0){MuX=MuX_PR;}       
else if(cosMuX<=0 && sinMuX>=0){MuX=MuX_PR;}
else if(cosMuX<=0 && sinMuX<=0){MuX=2*PI-MuX_PR;}
else if(cosMuX>=0 && sinMuX<=0){MuX=2*PI-MuX_PR;}

a0x=alphaX;
b0x=betaX;
mu_xTent=MuX;

output << "JDT:    MuX " <<    MuX << "\n";
double QX=MuX/2/PI;
output << "JDT:    QX  " <<    QX  << "\n";
output <<                             "\n";

double MY11=ry[1][1];double MY12=ry[1][2];
double MY21=ry[2][1];double MY22=ry[2][2];
double MYtr=MY11+MY22;
double cosMuY=MYtr/2;
if( (1-MYtr*MYtr/4)<0 ){std::cerr << "Y: Trying to take square root of a negative number!\n";exit(1);}
double betaY=abs(MY12)/sqrt(1-MYtr*MYtr/4);
double sinMuY=MY12/betaY;;
double alphaY=(MY11-MY22)/2/sinMuY;
output << "JDT: betaY  " << betaY  << "\n";
output << "JDT: cosMuY " << cosMuY << "\n";
output << "JDT: sinMuY " << sinMuY << "\n";
output << "JDT: alphaY " << alphaY << "\n";
double MuY_PR=acos(cosMuY);
double MuY;
if     (cosMuY>=0 && sinMuY>=0){MuY=MuY_PR;}
else if(cosMuY<=0 && sinMuY>=0){MuY=MuY_PR;}
else if(cosMuY<=0 && sinMuY<=0){MuY=2*PI-MuY_PR;}
else if(cosMuY>=0 && sinMuY<=0){MuY=2*PI-MuY_PR;}
a0y=alphaY;
b0y=betaY;
mu_yTent=MuY;

output << "JDT:    MuY " <<    MuY << "\n";
double QY=MuY/2/PI;
output << "JDT:    QY  " <<    QY  << "\n";
output <<                             "\n";
output.close();


 return 1;
}
