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
 if(argc!=4){
  std::cout << "usage: ./tracker ./data/E_FirstTest.sxf 30 -1.3 (> ! myOut)\n";
  std::cout << "argv[0] is this executable         - ./tracker\n";
  std::cout << "argv[1] is the input sxf file      - ./data/E_FirstTest.sxf\n";
  std::cout << "argv[2] is the nominal bend radius - 30      \n";
  std::cout << "argv[3] is the nominal electrode m - -1.3    \n";
  std::cout << "                                             \n";
  std::cout << "This radius is used to set the scale         \n";
  std::cout << "of the probe parameters.                     \n";
  std::cout << "It can be estimated from the sxf file(e.g.   \n";
  std::cout << "arc = 2.35619449019/                         \n";
  std::cout << "kl = 0.0785398163398 =                       \n";
  std::cout << "approximately 30).                           \n";
  std::cout << "It is a little subtle (e.g. injection issues,\n";
  std::cout << "manufacturing errors, setup errors, ...).    \n";
  std::cout << "A further subtlety is that angular           \n";
  std::cout << "momentum breaks the element-algorithm-probe  \n";
  std::cout << "paradigm, coupling probe parameter momentum  \n";
  std::cout << "with element parameter bend radius.          \n";
  std::cout << "#############################################\n";
  std::cout << "Nota bene: file simulatedProbeValues         \n";
  std::cout << "           is setup for post processing.     \n";
  std::cout << "           A single (1) turn is assumed.     \n";
  std::cout << "                                             \n";
  std::cout << "           It is intended to be edited       \n";
  std::cout << "           for specific user parameter       \n";
  std::cout << "           tracking.                         \n";
  std::cout << "                                             \n";
  std::cout << "           It has brief comments to this     \n";
  std::cout << "           effect.                           \n";
  std::cout << "#############################################\n";
  exit(0);
 }

  ofstream m_m;
  m_m.open ("m_m");
  m_m << argv[3];
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
/*
 double trtrout[5][9];
 for(int i=0;i<5;i++){
  for(int j=0;j<9;j++){
   trtrout[i][j]=0;
  }
 }
 double      rx[3][3];
 for(int i=0;i<3;i++){
  for(int j=0;j<3;j++){
        rx[i][j]=0;
  }
 }
*/

 // ************************************************************************
 std::cout << "\nDefine the space of Taylor maps." << std::endl;
 // ************************************************************************

  shell.setMapAttributes(UAL::Args() << UAL::Arg("order", order));
//shell.setMapAttributes(UAL::Args() << UAL::Arg("order", 5));

 // ************************************************************************
 std::cout << "\nBuild lattice." << std::endl;
 // ************************************************************************

 shell.readSXF(UAL::Args() << UAL::Arg("file",  sxfFile.c_str()));

 // ************************************************************************
 std::cout << "\nAdd split ." << std::endl;
 // ************************************************************************

  
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sbend")      << UAL::Arg("ir", split));
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Quadrupole") << UAL::Arg("ir", 0));
 shell.addSplit(UAL::Args() << UAL::Arg("lattice", "ring") << UAL::Arg("types", "Sextupole")  << UAL::Arg("ir", 0));

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

//#include "setBeamAttributes.hh"

// PAC::BeamAttributes& ba = shell.getBeamAttributes();

 // ************************************************************************
 std::cout << "\nLinear analysis." << std::endl;
 // ************************************************************************
  
/*
 // Make linear matrix

 std::cout << " matrix" << std::endl;
 shell.map(UAL::Args() << UAL::Arg("order", 1) << UAL::Arg("print", mapFile.c_str()));

 // Calculate twiss
  
 std::cout << " twiss (ring )" << std::endl;
 shell.twiss(UAL::Args() << UAL::Arg("print", twissFile.c_str()));

 std::cout << " calculate suml" << std::endl;
 shell.analysis(UAL::Args());
*/

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

 std::cout << "\n SXF_TRACKER tracker, ";
 std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

 // ************************************************************************
 std::cout << "\nBunch Part." << std::endl;
 // ************************************************************************

// ba.setG(1.7928474);             // adds proton G factor

/*
 PAC::Bunch bunch(1);               // bunch with 1 particle(s)
 bunch.setBeamAttributes(ba);

 PAC::Spin spin;
 spin.setSX(0.0);
 spin.setSY(0.0);
 spin.setSZ(1.0);

  bunch[0].getPosition().set(dx,dpx,dy,dpy,0,p5Input);
*/
/*
  bunch[0].getPosition().set(1.e-4,0.    ,1.e-4,0.    ,0.,0.);
  bunch[1].getPosition().set(0.   ,0.5e-5,0.   ,0.    ,0.,0.);
  bunch[2].getPosition().set(0.   ,0.    ,1.e-4,0.    ,0.,0.);
  bunch[3].getPosition().set(0.   ,0.    ,0.   ,0.5e-6,0.,0.);
*/

 // ************************************************************************
 std::cout << "\nTracking. " << std::endl;
 // ************************************************************************

 double t; // time variable

// int turns = 1024;

 positionPrinter pP;
 pP.open(orbitFile.c_str());
// xmgracePrint xP;
// xP.open("bunchSub0");

 ba.setElapsedTime(0.0);

 for(int iturn = 0; iturn <= (turns-1); iturn++){
//ap -> propagate(bunch);
  for(int ip=0; ip < bunch.size(); ip++){
   pP.write(iturn, ip, bunch);
// xP.write(iturn, ip, bunch);
  }
  ap -> propagate(bunch);
 }

 pP.close();
// xP.close();

/*
trtrout[1][1]=bunch[1].getPosition().getX();
trtrout[1][2]=bunch[2].getPosition().getX();
     rx[1][1]=(trtrout[1][1]-trtrout[1][2]);
std::cout << "TDJ-rx-DIFF" << "1 " << rx[1][1] << "\n";
     rx[1][1]=(trtrout[1][1]-trtrout[1][2])/2/x1typ;
std::cout << "TDJ-rx-RSLT" << "1 " << rx[1][1] << "\n";
*/
#include"trtrout"

double MX11=rx[1][1];double MX12=rx[1][2];
double MX21=rx[2][1];double MX22=rx[2][2];
double MXtr=MX11+MX22;
double cosMuX=MXtr/2;
double betaX=abs(MX12)/sqrt(1-MXtr*MXtr/4);
double sinMuX=MX12/betaX;;
double alphaX=(MX11-MX22)/2/sinMuX;
std::cout << "JDT: betaX  " << betaX  << "\n";
std::cout << "JDT: cosMuX " << cosMuX << "\n";
std::cout << "JDT: sinMuX " << sinMuX << "\n";
std::cout << "JDT: alphaX " << alphaX << "\n";
double MuX_PR=acos(cosMuX);
double MuX;
if     (cosMuX>=0 && sinMuX>=0){MuX=MuX_PR;}
else if(cosMuX<=0 && sinMuX>=0){MuX=MuX_PR;}
else if(cosMuX<=0 && sinMuX<=0){MuX=2*PI-MuX_PR;}
else if(cosMuX>=0 && sinMuX<=0){MuX=2*PI-MuX_PR;}
std::cout << "JDT:    MuX " <<    MuX << "\n";
double QX=MuX/2/PI;
std::cout << "JDT:    QX  " <<    QX  << "\n";
std::cout <<                             "\n";

double MY11=ry[1][1];double MY12=ry[1][2];
double MY21=ry[2][1];double MY22=ry[2][2];
double MYtr=MY11+MY22;
double cosMuY=MYtr/2;
double betaY=abs(MY12)/sqrt(1-MYtr*MYtr/4);
double sinMuY=MY12/betaY;;
double alphaY=(MY11-MY22)/2/sinMuY;
std::cout << "JDT: betaY  " << betaY  << "\n";
std::cout << "JDT: cosMuY " << cosMuY << "\n";
std::cout << "JDT: sinMuY " << sinMuY << "\n";
std::cout << "JDT: alphaY " << alphaY << "\n";
double MuY_PR=acos(cosMuY);
double MuY;
if     (cosMuY>=0 && sinMuY>=0){MuY=MuY_PR;}
else if(cosMuY<=0 && sinMuY>=0){MuY=MuY_PR;}
else if(cosMuY<=0 && sinMuY<=0){MuY=2*PI-MuY_PR;}
else if(cosMuY>=0 && sinMuY<=0){MuY=2*PI-MuY_PR;}
std::cout << "JDT:    MuY " <<    MuY << "\n";
double QY=MuY/2/PI;
std::cout << "JDT:    QY  " <<    QY  << "\n";
std::cout <<                             "\n";

      int ip=0;
      int iturn=0;
      PAC::Position& pos = bunch[ip].getPosition();

      double x  = pos.getX();
      double px = pos.getPX();
      double y  = pos.getY();
      double py = pos.getPY();
      double ct = pos.getCT();
      double de = pos.getDE();

      double wp_time = t0 + (-ct /UAL::clight );
      double ew      = 0;                         //de * p + energy;

      double psp0    = 0;                         //get_psp0(pos, v0byc);

      char endLine = '\0';
      char line1[200];
    
      sprintf(line1, "%1d %7d    %-15.9e %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %-15.10e %-15.10e %c",
              ip, iturn, wp_time, x, px, y, py, ct, de, psp0, ew, endLine);

//    std::cout << line1 << std::endl;

 return 1;
}
