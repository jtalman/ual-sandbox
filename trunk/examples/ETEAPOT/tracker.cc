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

//#include "timer.h"
#include "positionPrinter.hh"

//#include "globalBlock.h"

#include "ETEAPOT/Integrator/DipoleTracker.hh"

using namespace UAL;

int main(int argc,char * argv[]){
 if(argc!=2){
  std::cout << "usage: ./tracker ./data/pre-E_pEDm.sxf (> ! myOut)\n";
  std::cout << "argv[0] is this executable: ./tracker\n";
  std::cout << "argv[1] is the input sxf file - ./data/pre-E_pEDm.sxf\n";
  exit(0);
 }

std::string mysxf    =argv[1];
std::string mysxfbase=mysxf.substr(7,mysxf.size()-11);
std::cout << "mysxf     " << mysxf.c_str() << "\n";
std::cout << "mysxfbase " << mysxfbase.c_str() << "\n";

#include "designBeamValues.hh"
#include "extractParameters.h"

 UAL::Shell shell;

/*
 string fname;
 cout << "Enter a file name: ";
 getline(cin, fname);
*/
 ofstream ofstrm("flat_sxf");
 if (!ofstrm){
   cout << "Couldn’t open file: " << "flat_sxf" << endl;
 }
 else{
  cout << "Found and opened file: " << "flat_sxf" << endl;
 }
 ofstrm << setiosflags( ios::showpos   );
 ofstrm << setiosflags( ios::uppercase );
 ofstrm << setiosflags( ios::scientific );
 ofstrm << setw( 11 );
 ofstrm << setfill( ' ' );
 ofstrm << setiosflags( ios::left );
 ofstrm << setprecision(4) ;

 // ************************************************************************
 std::cout << "\nDefine the space of Taylor maps." << std::endl;
 // ************************************************************************

 shell.setMapAttributes(UAL::Args() << UAL::Arg("order", 5));

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

// shell.setBeamAttributes(UAL::Args() << UAL::Arg("energy", e0) << UAL::Arg("mass", m0));
// shell.setBeamAttributes(UAL::Args() << UAL::Arg("elapsedTime", 0));
#include "setBeamAttributes.hh"

 PAC::BeamAttributes& ba = shell.getBeamAttributes();

 // ************************************************************************
 std::cout << "\nLinear analysis." << std::endl;
 // ************************************************************************
  
 // Make linear matrix

 std::cout << " matrix" << std::endl;
 shell.map(UAL::Args() << UAL::Arg("order", 1) << UAL::Arg("print", mapFile.c_str()));

 // Calculate twiss
  
 std::cout << " twiss (ring )" << std::endl;
 shell.twiss(UAL::Args() << UAL::Arg("print", twissFile.c_str()));

 OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();
 Teapot* teapot = optics.m_teapot;
  
 PacSurveyData surveyData;
 char drift[13]="drift";
 std::string nameInput;
 std::string nameOutput;
 double xN = 0;
 double yN = 0;
 double zN = 0;

 double xX;
 double yX;
 double zX;

 double a,b,c,d;
 double f;                                     // "alpha" -->> fuh
 double u;                                     // "beta"  -->> uh
 double g;                                     // gamma

 double AN,BN,CN;                              // "quadratic coefficients" via (a,0,b) element entry point
                                               // AN x^2 + BN x + CN = 0
 double AX,BX,CX;                              // AX x^2 + BX x + CX = 0   via (c,0,d) element  exit point

 double xCPN,yCPN=0,zCPN;                         // center via plus  quadratic solution and element entry point
 double xCPX,yCPX=0,zCPX;                         // center via plus  quadratic solution and element exit  point
 double xCMN,yCMN=0,zCMN;                         // center via minus quadratic solution and element entry point
 double xCMX,yCMX=0,zCMX;                         // center via minus quadratic solution and element exit  point

 int myIndex=-1;

 for(int i = 0; i < teapot->size(); i++){
  TeapotElement& te = teapot->element(i);
  nameInput=te.getDesignName();
  if(nameInput=="mkb" ){nameInput+="  "; }
  if(nameInput=="mk1" ){nameInput+="  "; }
  if(nameInput=="mke" ){nameInput+="  "; }
  if(nameInput=="bpm" ){nameInput+="  "; }
  if(nameInput=="q1h" ){nameInput+="  "; }
  if(nameInput=="q1"  ){nameInput+="   ";}
  if(nameInput=="q2"  ){nameInput+="   ";}
  if(nameInput=="q3h" ){nameInput+="  "; }
  if(nameInput=="s1"  ){nameInput+="   ";}
  if(nameInput=="s2"  ){nameInput+="   ";}
  if(nameInput=="s3"  ){nameInput+="   ";}
  if(nameInput=="bend"){nameInput+=" ";  }

  myIndex++;

  if(nameInput.size()>=1){
   nameOutput=nameInput;
  }
  else{
   nameOutput=drift;
  }

  std::remove(nameOutput.begin(), nameOutput.end(), ' ');
 
  teapot->survey(surveyData,i,i+1);

  xX = surveyData.survey().x();
  yX = surveyData.survey().y();
  zX = surveyData.survey().z();

  a=xN,b=zN;
  c=xX   ,d=zX   ;
  f=c-a,  u=d-b;
  g=-(a*a+b*b-c*c-d*d)/2;

  AN=1+f*f/u/u;
  BN=-2*(a+g*f/u/u-f/u);                                   // ANx^2 + BNx + CN = 0
  CN=a*a+g*g/u/u-2*g/u+b*b-R0*R0;                          // via entry point

  AX=1+f*f/u/u;
  BX=-2*(c+g*f/u/u-f/u);                                   // AXx^2 + BXx + CX = 0
  CX=c*c+g*g/u/u-2*g/u+d*d-R0*R0;                          // via  exit point

  xCPN=(-2*BN+sqrt(BN*BN-4*AN*CN))/2/AN;  zCPN=(g-xCPN*f)/u;  // plus  quadratic solution via entry point
  xCMN=(-2*BN-sqrt(BN*BN-4*AN*CN))/2/AN;  zCMN=(g-xCMN*f)/u;  // minus quadratic solution via entry point

  xCPX=(-2*BX+sqrt(BX*BX-4*AX*CX))/2/AX;  zCPX=(g-xCPX*f)/u;  // plus  quadratic solution via  exit point
  xCMX=(-2*BX-sqrt(BX*BX-4*AX*CX))/2/AX;  zCMX=(g-xCMX*f)/u;  // minus quadratic solution via  exit point

  ofstrm
         << setw( 5 )
         << i << " "
         << nameOutput << ": ["
/*  
         << setw( 11 )
         << xN << ", "
         << setw( 11 )
         << yN << ", "
         << setw( 11 )
         << zN << ")"
         << " to ["
         << setw( 11 )
         << xX << ", "
         << setw( 11 )
         << yX << ", "
         << setw( 11 )
         << zX << ") "
*/  

//       << "(" << xCPN << "," << zCPN << ") "
         << "(" << xCMN << "," << zCMN << ") "
//       << " " << sqrt((xCPN-a)*(xCPN-a)+(zCPN-b)*(zCPN-b))
//       << "(" << xCPX << "," << zCPX << ") "
         << "(" << xCMX << "," << zCMX << ") "
         << sqrt((xCMN-a)*(xCMN-a)+(zCMN-b)*(zCMN-b)) << " "
         << sqrt((xCMN-c)*(xCMN-c)+(zCMN-d)*(zCMN-d)) << " "
         << sqrt((xCMX-a)*(xCMX-a)+(zCMX-b)*(zCMX-b)) << " "
         << sqrt((xCMX-c)*(xCMX-c)+(zCMX-d)*(zCMX-d))
//       << " (" << xCPN << "," << zCPN << ") "
//       << " (" << xCMX << "," << zCMX << ") "

         << std::endl;

  xN=xX;
  yN=yX;
  zN=zX;
 }

 double suml = surveyData.survey().suml();
  
 char endLine ='\0';
 double twopi=2*PI;
 char line[200];
 double at=0;
 for(int i = 0; i < teapot->size(); i++){
  at += teapot->element(i).l();
  sprintf(line, "%5d %15.7e %-10s%c", 
   i, at, 
   teapot->element(i).getDesignName().c_str(), endLine);
  std::cout << line << std::endl;

 }

 std::cout << " calculate suml" << std::endl;
 shell.analysis(UAL::Args());

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

 PAC::Bunch bunch(1);               // bunch with one particle
 bunch.setBeamAttributes(ba);

 PAC::Spin spin;
 spin.setSX(0.0);
 spin.setSY(0.0);
 spin.setSZ(1.0);

 std::cout << "probeEscr0 " << probeEscr0 << "\n";

 for(int ip=0; ip < bunch.size(); ip ++){
  bunch[ip].getPosition().set(probe__dx0,probe_dpx0,probe__dy0,probe_dpy0,probe_cdt0,probeEscr0);
  bunch[ip].setSpin(spin);
 }

 // ************************************************************************
 std::cout << "\nTracking. " << std::endl;
 // ************************************************************************

 double t; // time variable

 int turns = 1;

 positionPrinter pP;
 pP.open(orbitFile.c_str());

 ba.setElapsedTime(0.0);

 for(int iturn = 1; iturn <= turns; iturn++){
  ap -> propagate(bunch);
  for(int ip=0; ip < bunch.size(); ip++){
   pP.write(iturn, ip, bunch);
  }
 }
 pP.close();
 return 1;
}
