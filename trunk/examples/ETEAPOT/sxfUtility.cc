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
 ofstrm << setiosflags( ios::right );
 ofstrm << setprecision(4) ;

 UAL::Shell shell;
 std::string sxfFile = argv[1];
 shell.readSXF(UAL::Args() << UAL::Arg("file",  sxfFile.c_str()));
 shell.use(UAL::Args() << UAL::Arg("lattice", "ring"));

 OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();
 Teapot* teapot = optics.m_teapot;

 double R0=40;

 PacSurveyData surveyData;
 char drift[13]="drift      ";
 std::string nameInput;
 std::string nameOutput;
 double xN = 0;
 double yN = 0;
 double zN = 0;
 double sN = 0;

 double xX;
 double yX;
 double zX;
 double sX;

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
/*
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
*/
  if(nameInput=="mbegin"){nameInput+="     "; }
  if(nameInput=="marcin"){nameInput+="     "; }
  if(nameInput=="mcellin"){nameInput+="    "; }
  if(nameInput=="sb"){nameInput+="         "; }
  if(nameInput=="qb"){nameInput+="         "; }
  if(nameInput=="bh"){nameInput+="         "; }
  if(nameInput=="sc"){nameInput+="         "; }
  if(nameInput=="qc"){nameInput+="         "; }
  if(nameInput=="qa"){nameInput+="         "; }
  if(nameInput=="sa"){nameInput+="         "; }
  if(nameInput=="mcellcenter"){nameInput+=""; }
  if(nameInput=="mcellout"){nameInput+="   "; }
  if(nameInput=="marcout"){nameInput+="    "; }
  if(nameInput=="mend"){nameInput+="       "; }

  myIndex++;

  if(nameInput.size()>=1){
   nameOutput=nameInput;
  }
  else{
   nameOutput=drift;
  }

//std::remove(nameOutput.begin(), nameOutput.end(), ' ');
 
  teapot->survey(surveyData,i,i+1);

  xX = surveyData.survey().x();
  yX = surveyData.survey().y();
  zX = surveyData.survey().z();
  sX = surveyData.survey().suml();

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
         << setw( 11 )
         << nameOutput << " "
         << setw( 11 )
         << sN << " ( "
    
         << setw( 11 )
         << xN << " , "
//       << xX << " "
         << setw( 11 )
         << yN << " , "
//       << yX << " "
         << setw( 11 )
         << zN << " )"
//       << zX << " "
//       << nameOutput << " "
//       << setw( 11 )
//       << sN
//       << sX
         << "-( "
         << setw( 11 )
         << xX << " , "
         << setw( 11 )
         << yX << " , "
         << setw( 11 )
         << zX << " ) "
         << sX << " "
         << sX-sN
         << std::endl;
    
  xN=xX;
  yN=yX;
  zN=zX;
  sN=sX;
 }

 return 1;
}
