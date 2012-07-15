// Library     : Eteapot
// File        : Main/Eteapot.cc
// Copyright   : see Copyright file
// Author      : John Talman

#include <stdio.h>
#include <time.h>
#include <cmath>
#include <fstream>

#include "Main/Eteapot.h"

#include "UAL/APDF/APDF_Builder.hh"

#include "positionPrinter.hh"

//#include "UAL/UI/Shell.hh"

// Public methods

// Commands

// Tracking

  void Eteapot::twissFromTracking( PAC::BeamAttributes ba, UAL::AcceleratorPropagator* ap, float m_m, double& a0x,double& b0x,double& mu_xTent,double& a0y,double& b0y,double& mu_yTent )
//void Eteapot::twissFromTracking( PAC::BeamAttributes ba, UAL::AcceleratorPropagator* ap, float m_m, double& a0x,double& b0x,double& a0y,double& b0y )
//void Eteapot::twissFromTracking( PAC::BeamAttributes ba, UAL::AcceleratorPropagator* ap, float m_m, double a0x,double b0x,double a0y,double b0y )
//void Eteapot::twissFromTracking( PAC::BeamAttributes ba, UAL::AcceleratorPropagator* ap, float m_m )

//void Eteapot::twissFromTracking( PAC::BeamAttributes ba, UAL::AcceleratorPropagator* ap, std::string filename )
{ 
#ifdef lngTrmTrk
 std::cerr << "################ TWISS NOT ALLOWED ################\n";
 exit(0);
#endif
 std::cout << "JDT: file " << __FILE__ << " line " <<__LINE__ << " enter void Eteapot::twissFromTracking()\n";

/*
 UAL::Shell shell;
 shell.setMapAttributes(UAL::Args() << UAL::Arg("order", order));
 std::string bs="/";
 std::string ps =path+bs+sxfFile;
 std::cout << "ps.c_str() " << ps.c_str() << "\n";
 shell.readSXF(UAL::Args() << UAL::Arg("file",  ps.c_str()));
*/

 std::cout << "ap->getName() " << ap->getName() << "\n";
 std::cout << "\n SXF_TRACKER tracker, ";
 std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

int turns=1;
#include "probeDataForTwiss"
#include "trtrin"
bunch[ 0].getPosition().set( 0, 0, 0, 0, 0, 0 );

positionPrinter pP;
pP.open("TBT");

//int iturn = 0;
//for(iturn = 0; iturn <= (turns-1); iturn++){
 for(int ip=0; ip < bunch.size(); ip++){
  pP.write(0, ip, bunch);
 }
 ap -> propagate(bunch);
//}
//pP.write(iturn, ip, bunch);
 for(int ip=0; ip < bunch.size(); ip++){
  pP.write(1, ip, bunch);
 }

pP.close();

std::ofstream output;
char buffr2 [10];
sprintf(buffr2,"%+5.2f",m_m);
std::string bp2(buffr2);
std::string sT = "out/STT/TWISS_m=";
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

 std::cout << "JDT: file " << __FILE__ << " line " <<__LINE__ << " leave void Eteapot::twissFromTracking()\n";
}
