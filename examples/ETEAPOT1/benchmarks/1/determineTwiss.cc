#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

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
#include "ETEAPOT/Integrator/MltTracker.hh"
//#include "ETEAPOT/Integrator/DriftTracker.hh"
#include "ETEAPOT/Integrator/MarkerTracker.hh"
#include "ETEAPOT/Integrator/RFCavityTracker.hh"

using namespace UAL;

int main(int argc,char * argv[]){
// std::cerr << "ETEAPOT::DipoleTracker::m_m " << ETEAPOT::DipoleTracker::m_m << "\n";
// std::cerr << "ETEAPOT::MltTracker::m_m    " << ETEAPOT::MltTracker::m_m    << "\n";

 if(argc!=4){
  std::cout << "usage: ./determineTwiss ./data/E_BM_M1.0_sl4.sxf -1 40 (>&! OUT)\n";
  std::cout << "argv[0] is this executable         - ./determineTwiss           \n";
  std::cout << "argv[1] is the input sxf file      - ./data/E_BM_M1.0_sl4.sxf   \n";
  std::cout << "argv[2] is the nominal electrode m - +1                         \n";
  std::cout << "argv[3] is the nominal electrode bend radius - 40=.7854/.0196   \n";
  exit(0);
 }

 ofstream m_m;
 m_m.open ("m_m");
 m_m << argv[2];
 m_m.close();
 ETEAPOT::DipoleTracker* edt=new ETEAPOT::DipoleTracker();
 ETEAPOT::MltTracker*    mdt=new ETEAPOT::MltTracker();
 std::cerr << "ETEAPOT::DipoleTracker::m_m " << ETEAPOT::DipoleTracker::m_m << "\n";
 std::cerr << "ETEAPOT::MltTracker::m_m    " << ETEAPOT::MltTracker::m_m    << "\n";
               ETEAPOT::DipoleTracker::m_m=atof( argv[2] );
               ETEAPOT::MltTracker::m_m   =atof( argv[2] );
 std::cerr << "ETEAPOT::DipoleTracker::m_m " << ETEAPOT::DipoleTracker::m_m << "\n";
 std::cerr << "ETEAPOT::MltTracker::m_m    " << ETEAPOT::MltTracker::m_m    << "\n";

 std::string mysxf    =argv[1];
 std::string mysxfbase=mysxf.substr(7,mysxf.size()-11);
 std::cout << "mysxf     " << mysxf.c_str() << "\n";
 std::cout << "mysxfbase " << mysxfbase.c_str() << "\n";

 UAL::Shell shell;

 #include "designBeamValues.hh"
 #include "setBeamAttributes.hh"
 PAC::BeamAttributes& ba = shell.getBeamAttributes();
 #include "extractParameters.h"
// #include "probeDataForTwiss"
#define TINY 1E-6
double x1typ  = TINY;
double x2typ  = TINY;
double y1typ  = TINY, x3typ  = TINY;
double y2typ  = TINY, x4typ  = TINY;
double x5typ  = TINY;
double deltyp = TINY, x6typ  = TINY;
 #include "trtrin"
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

#include"forTwissStndrdPrtclQuiet"
//#include"forTwissStndrdPrtclVerbose"

 double a0x=     (double)0;
 double b0x=     (double)0;
 double mu_xTent=(double)0;
 double a0y=     (double)0;
 double b0y=     (double)0;
 double mu_yTent=(double)0;
//int turns=1;
 turns=1;
//#include "probeDataForTwiss"
 std::cerr << "ETEAPOT::DipoleTracker::m_m " << ETEAPOT::DipoleTracker::m_m << "\n";
 std::cerr << "ETEAPOT::MltTracker::m_m    " << ETEAPOT::MltTracker::m_m    << "\n";

//#include "spin"
 ofstream iS;
 iS.open ("initialSpin");    // for server side compatibility
                             // Full spin functionality is anticipated in 
                             // orbitsWithSpin.cc or such
 iS.close();

 ap -> propagate(bunch);

 char buffr2 [10];
 sprintf( buffr2,"%+5.2f",atof(argv[2]) );
 std::string bp2(buffr2);
 std::string sT = "out/TWISS/TWISS_m=";
 sT+=bp2;

//std::cerr.open( filename.c_str() );
//std::cerr.open("TWISS");
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
 std::cerr << "JDT: betaX  " << betaX  << "\n";
 std::cerr << "JDT: cosMuX " << cosMuX << "\n";
 std::cerr << "JDT: sinMuX " << sinMuX << "\n";
 std::cerr << "JDT: alphaX " << alphaX << "\n";
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

 std::cerr << "JDT:    MuX " <<    MuX << "\n";
 double QX=MuX/2/PI;
 std::cerr << "JDT:    QX  " <<    QX  << "\n";
 std::cerr <<                             "\n";

 double MY11=ry[1][1];double MY12=ry[1][2];
 double MY21=ry[2][1];double MY22=ry[2][2];
 double MYtr=MY11+MY22;
 double cosMuY=MYtr/2;
 if( (1-MYtr*MYtr/4)<0 ){std::cerr << "Y: Trying to take square root of a negative number!\n";exit(1);}
 double betaY=abs(MY12)/sqrt(1-MYtr*MYtr/4);
 double sinMuY=MY12/betaY;;
 double alphaY=(MY11-MY22)/2/sinMuY;
 std::cerr << "JDT: betaY  " << betaY  << "\n";
 std::cerr << "JDT: cosMuY " << cosMuY << "\n";
 std::cerr << "JDT: sinMuY " << sinMuY << "\n";
 std::cerr << "JDT: alphaY " << alphaY << "\n";
 double MuY_PR=acos(cosMuY);
 double MuY;
 if     (cosMuY>=0 && sinMuY>=0){MuY=MuY_PR;}
 else if(cosMuY<=0 && sinMuY>=0){MuY=MuY_PR;}
 else if(cosMuY<=0 && sinMuY<=0){MuY=2*PI-MuY_PR;}
 else if(cosMuY>=0 && sinMuY<=0){MuY=2*PI-MuY_PR;}
 a0y=alphaY;
 b0y=betaY;
 mu_yTent=MuY;

 std::cerr << "JDT:    MuY " <<    MuY << "\n";
 double QY=MuY/2/PI;
 std::cerr << "JDT:    QY  " <<    QY  << "\n";
 std::cerr <<                             "\n";

 std::cerr << "ETEAPOT::DipoleTracker::m_m " << ETEAPOT::DipoleTracker::m_m << "\n";
 std::cerr << "ETEAPOT::MltTracker::m_m    " << ETEAPOT::MltTracker::m_m    << "\n";

 std::cerr << "./transferMatrices " << ETEAPOT::DipoleTracker::m_m << " " << alphaX << " " << betaX << " " << alphaY << " " << betaY << " " << nonDrifts << ">! betaFunctions\n";
 return (int)0;
}
