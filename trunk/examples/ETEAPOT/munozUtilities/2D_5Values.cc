#include<iostream>
#include<fstream>

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"../clr"
#include"../../../codes/UAL/src/UAL/Common/Def.hh"

#define GAMMA_FROZEN_SPIN 1.248107349
long double gFS=UAL::pFSG;

int main(int argc,char*argv[]){
 long double mpG=UAL::pmass;
 long double ep=UAL::elemCharge;
 long double c=UAL::clight;
 long double vP=UAL::vcmPerm;
 long double kC=1.L/4.L/UAL::pi/vP;

 if(argc!=10){
  printf("This program does 3 things:\n");
  printf("      1) Calculates off momentum circle quantities for rD + dr = rDelta\n");
  printf("      2) Creates a file to be plotted to visually verify the circle\n");
  printf("      3) Calculates absolute (not deviation) time of flight for the near circle\n");
  printf("See the end of file runExample for a little more detail\n");
  printf("Usage: ./2D_5Values kD(Design) rD(esign) gD(esign) dr0 th0 thD0 rD0 dg0 Ngrid\n");
  exit(1);
 }

 long double oneE9=1.e+9;
 long double mp=mpG*ep*oneE9/c/c;
 long double Ep=mp*c*c;

// system constants
 long double kD=atof(argv[1]);
 long double rD=atof(argv[2]); 
 long double gD=atof(argv[3]); 

 long double bD=sqrt(1.L-1.L/gD/gD);
 long double vD=bD*c;
 long double pD=gD*mp*vD;
 long double LD=rD*pD;

 long double LC=kD/c;
// system constants

// initial variables
 long double dr0=atof(argv[4]);
 long double dx0=dr0;
 long double x0=rD+dr0;
 long double th0=atof(argv[5]);
 long double thD0=atof(argv[6]);
 long double rD0=atof(argv[7]);
 long double dg0=atof(argv[8]);
// initial variables

 printf(RESET);
 printf("\nPhysical Constants\n");
 printf("_________________________________\n");
 printf("mp, proton mass   %+25.15Le [kg]\n",mp);
 printf("Ep (mp * c * c)   %+25.15Le [J]\n",Ep);
 printf("ep, proton chge   %+25.15Le [C] - also designated q below\n",ep);
 printf("c, speed light    %+25.15Le [m/s]\n",c);
 printf("vP, vacuum perm   %+25.15Le [F/m]\n",vP);
 printf("kC, Coulomb       %+25.15Le [m/F]\n",kC);
 printf("_________________________________\n");
 printf("\n");

 printf("\nSystem Constants - D for Design\n");
 printf("_________________________________\n");
 printf("kD, Munoz         %+25.15Le [J*m]\n",kD);
 printf("rD, rds dsgn      %+25.15Le [m]\n",rD);
 printf("gD, gamma dsgn    %+25.15Le []\n",gD);
 printf("\n");
 printf("bD, beta dsgn     %+25.15Le []\n",bD);
 printf("vD, vlcty dsgn    %+25.15Le [m/s]\n",vD);
 printf("pD, momen dsgn    %+25.15Le [kg*m/s]\n",pD);
 printf("LD, ang mom dsgn  %+25.15Le [kg*m*m/s]\n",LD);
 printf("LC, ang mom crtcl %+25.15Le [kg*m*m/s] - kD/c\n",LC); 
 printf("LD/LC             %+25.15Le []\n",LD/LC); 
 printf("_________________________________\n");
 printf("\n");

 printf("\nInitial Probe Parameter Values \"...0\"\n");
 printf("_________________________________\n");
 printf("dr0               %+25.15Le [m]\n",dr0);
 printf("dx0               %+25.15Le [m]\n",dx0);
 printf("x0                %+25.15Le [m]\n",x0);
 printf("th0               %+25.15Le []\n",th0);
 printf("thD0              %+25.15Le [1/s]\n",thD0);
 printf("rD0               %+25.15Le [m/s]\n",rD0);
 printf("dg0               %+25.15Le []\n",dg0); 
 printf("_________________________________\n");
 printf("\n");

 printf("\nConstraints\n");
 printf("_________________________________\n");
 long double ElD=kD/ep/rD/rD;
 long double ERT=1.1710642;//1.171064565;       // Escr0 page 14 ETEAPOT-expanded.pdf
 long double ElRT=10.48270839e+6;
 long double kRT=ElRT*ep*rD*rD;
 long double kMCnstrn=mp*rD*(gD-1.L/gD)*c*c;
 long double QM=-ElD*rD*rD/kC; 
 long double kMCnstrn2=kC*QM*ep;
   long double q=ep; 
// long double q=ep; 
 printf("ElD, elc fld dsgn %+25.15Le [V/m]\n",ElD);
 printf("ElRT              %+25.15Le [V/m]\n",ElRT);
 printf("kRT               %+25.15Le [J*m]\n",kRT);
 printf("kD                %+25.15Le [J*m]\n",kD);
 printf("kMCnstrn          %+25.15Le [J*m] - this should equal k Munoz\n",kMCnstrn); 
 printf("QM, fixed charge  %+25.15Le [C]\n",QM); 
 printf("kC QM q/rD/rD     %+25.15Le [Nt] - should equal below\n",kC*QM*q/rD/rD); 
 printf("q ElD             %+25.15Le [Nt] - should equal above\n",q*ElD); 
 printf("kMCnstrn2         %+25.15Le [Jm] - should equal argv[1]\n",kMCnstrn2); 
 printf("_________________________________\n");
 printf("\n");

 long double rDelta=rD+dr0;

 printf("Theoretical Delta Values (Input Radius Treated As Circle)\n");
 printf("_________________________________\n");
 long double aQ         = 1.L;
 long double bQ         = (kRT/mp/rDelta/rDelta/c)*(kRT/mp/rDelta/rDelta/c);
 long double cQ         = -(kRT/mp/rDelta/rDelta/rDelta)*(kRT/mp/rDelta/rDelta/rDelta);
 long double TheorTDDelta_sq= (-bQ + sqrt(bQ*bQ - 4.L*aQ*cQ))/2.L/aQ;
 long double TheorTDDelta=sqrt(TheorTDDelta_sq);
 long double TheorVDelta    = rDelta*TheorTDDelta;
 long double TheorBDelta    = TheorVDelta/c;
 long double TheorGDelta    = 1.L/sqrt(1.L-TheorVDelta*TheorVDelta/c/c);
 long double TheorPDelta    = TheorGDelta * mp * TheorVDelta;
 long double TheorPDeltac   = TheorPDelta*c;
// long double q=ep; 
 long double GeVperJ=1.L/q/oneE9;
 long double TheorPDeltacEV = TheorPDeltac*GeVperJ;
 long double TheorMEDeltaEV = sqrt(TheorPDeltacEV*TheorPDeltacEV+mpG*mpG);
 long double TheorPEDeltaEV = (-kD/rDelta+kD/rD)*GeVperJ;
 long double TheorTEDeltaEV = TheorMEDeltaEV+TheorPEDeltaEV;
 printf("rD, Design Radius   %+25.15Le [m]\n",rD);
 printf("rDelta Input Radius %+25.15Le [m]\n",rDelta);
 printf("TheorTDDelta        %+25.15Le []\n",TheorTDDelta); 
 printf("TheorVDelta         %+25.15Le [m/s]\n",TheorVDelta); 
 printf("TheorBDelta         %+25.15Le []\n",TheorBDelta); 
 printf("TheorGDelta         %+25.15Le []\n",TheorGDelta); 
 printf("TheorPDelta         %+25.15Le [kg m / s]\n",TheorPDelta); 
 printf("TheorPDeltac        %+25.15Le [J]\n",TheorPDeltac); 
 printf("TheorPDeltacEV      %+25.15Le [GeV]\n",TheorPDeltacEV); 
 printf("TheorMEDeltaEV      %+25.15Le [GeV]\n",TheorMEDeltaEV); 
 printf("TheorPEDeltaEV      %+25.15Le [GeV]\n",TheorPEDeltaEV); 
 printf("TheorTEDeltaEV      %+25.15Le [GeV]\n",TheorTEDeltaEV); 
 printf("_________________________________\n");
 printf("\n");

 long double g0=gD+dg0;
 long double g=g0;
 long double r0=rD+dr0;
 long double r=r0;

 long double L0=g0*mp*r0*r0*thD0;
 long double L=L0;

 long double RE_Mz=mp*c*c;

 long double MEM0=g*mp*c*c;
 long double MEM=MEM0;

 long double KE_Mz0=MEM0-mp*c*c;
 long double KE_Mz=KE_Mz0;

 long double PE_Mz0=-kRT/r;
 long double PE_Mz=PE_Mz0;

 long double EM0=MEM0+PE_Mz0;

 long double TE_Mz=RE_Mz+KE_Mz+PE_Mz;

 long double PE_cnv0=PE_Mz0+kRT/rD;
 long double PE_cnv=PE_cnv0;
 long double TE_cnv=MEM0+PE_cnv0;
// long double GeVperJ=1.L/q/1.Le9;
 long double EM=EM0;
/*
 printf("\nDynamic Constants (Conserved Quantities and Their Components)\n");
 printf("_________________________________\n");
 printf("r,  initial radius           %+25.15Le [m]\n",r); 
 printf("r0, initial radius           %+25.15Le [m]\n",r0); 
 printf("rD, design  radius           %+25.15Le [m]\n",rD); 
 printf("L=L0, cnsrv ng mm            %+25.15Le [kg*m*m/s]\n",L0); 
 printf("\n");
 printf("RE_Mz Rest Energy              %+25.15Le [J]\n",RE_Mz); 
 printf("KE_Mz=KE_Mz0                     %+25.15Le [J]\n",KE_Mz0); 
 printf("PE_Mz=PE_Mz0                     %+25.15Le [J]\n",PE_Mz0); 
 printf("RE_Mz+KE_Mz+PE_Mz                  %+25.15Le [J]\n",RE_Mz+KE_Mz+PE_Mz); 
 printf("\n");
 printf("RE_Mz*GeVperJ                  %+25.15Le [GeV]\n",RE_Mz*GeVperJ); 
 printf("KE_Mz*GeVperJ                  %+25.15Le [GeV]\n",KE_Mz0*GeVperJ); 
 printf("PE_Mz*GeVperJ                  %+25.15Le [GeV]\n",PE_Mz0*GeVperJ); 
 printf("PE_cnv*GeVperJ                   %+25.15Le [GeV]\n",PE_cnv*GeVperJ); 
 printf("kRT*GeVperJ/rD               %+25.15Le [GeV]\n",kRT*GeVperJ/rD); 
 printf("(RE_Mz+KE_Mz+PE_Mz)*GeVperJ        %+25.15Le [GeV]\n",(RE_Mz+KE_Mz+PE_Mz)*GeVperJ); 
 printf("(RE_Mz+KE_Mz+PE_cnv)*GeVperJ         %+25.15Le [GeV]\n",(RE_Mz+KE_Mz+PE_cnv)*GeVperJ); 
 printf("(RE_Mz+KE_Mz+PE_Mz+kRT/rD)*GeVperJ %+25.15Le [GeV]\n",(RE_Mz+KE_Mz+PE_Mz+kRT/rD)*GeVperJ); 
 printf("\n");
 printf("EM=EM0, cnsrv ngy            %+25.15Le [J]\n",EM0); 
 printf("EC=TE_cnv, cnsrv ngy            %+25.15Le [J]\n",TE_cnv); 
 printf("GeVperJ                      %+25.15Le []\n",GeVperJ); 
 printf("TE_cnv*GeVperJ                  %+25.15Le [GeV]\n",TE_cnv*GeVperJ); 
 printf("EM0*GeVperJ                  %+25.15Le [GeV]\n",EM0*GeVperJ); 
 printf("Ep/gD                        %+25.15Le [J]\n",Ep/gD); 
 printf("_________________________________\n");
 printf("\n");
*/

 printf("\nStarting Dynamic Variables\n");
 printf("_________________________________\n");
 printf("g, gamma          %+25.15Le [] - \"g0\"\n",g0); 
 printf("r0, radius        %+25.15Le [m] - \"r0\"\n",r0); 

 long double LcOver_k=L*c/kD;
 long double Efac=EM/Ep;
 long double kapSqu  = 1.L-1.L/LcOver_k/LcOver_k;
 long double kap=sqrt(kapSqu);
 long double lambda=kapSqu*L0*L0*c*c/kD/EM;   //   40;
 long double eps=0.;
 if(1.L-kapSqu/Efac/Efac>0){
  eps=LcOver_k*sqrt(1.L-kapSqu/Efac/Efac);
 }
 else{
//printf("\"d\" 1.L-kapSqu/Efac/Efac %+25.15Le\n",1.L-kapSqu/Efac/Efac);
 }
// long double r0=lambda/(1.L+eps);                                 <<<---   implement consistency check?
 printf("\n");
 printf("v0 (r0*thD0)      %+25.15Le [m/s]\n",r0*thD0); 
 printf("_________________________________\n");
 printf("\n");

 long double b=1.L/Efac/Efac-1.L;
 long double a = 1.L/LcOver_k;
 long double eps2=0.L;
 if(a*a-b+b*a*a > 0){
  eps2=1.L/a*sqrt( a*a - b + b*a*a );
 }
 else{
//printf("\"d\" a*a - b + b*a*a %+25.15Le\n",a*a - b + b*a*a);
 }
 long double a2=b/sqrt(1.L+b);

  printf("\n" "\033[1m\033[31m" "PURE MUNOZ CROSS CHECK\n");
//printf("\n" "\033[1m\033[30m" "PURE MUNOZ CROSS CHECK\n");
 printf("_________________________________\n");
 printf("*********************************\n");
 printf("lambda            %+25.15Le [m] - XXXXXX just for completeness, lambda plays no role here XXXXXX\n",lambda); 
 printf("LcOver_k          %+25.15Le []\n",LcOver_k); 
 printf("Efac              %+25.15Le []\n",Efac); 
 printf("Efac^2            %+25.15Le []\n",Efac*Efac); 
 printf("kapSqu            %+25.15Le []\n",kapSqu); 
 printf("Efac*Efac-kapSqu  %+25.15Le []\n",Efac*Efac-kapSqu); 
 printf("eps               %+25.15Le [] (\"d\" 1.L-kapSqu/Efac/Efac %+25.15Le)\n",eps,1.L-kapSqu/Efac/Efac); 
 printf("eps2              %+25.15Le [] (\"d\" a*a - b + b*a*a     %+25.15Le)\n",eps2,a*a - b + b*a*a); 
 printf("_________________________________\n");
 printf("*********************************\n");
 printf("\n");
 printf(RESET);

 long double A=g0*r0*thD0-kD*g0/L0;
 printf("A                 %+25.15Le [m/s]\n",A); 
 long double C=L0*mp*c*c/kD/EM0;
 printf("C                 %+25.15Le [s/m]\n",C); 
 long double AC=A*C;
 printf("AC                %+25.15Le []\n",AC); 
 printf("\n");

 long double B=-kap*g0*rD0;
 printf("B                 %+25.15Le [m/s]\n",B); 
 long double BC=B*C;
 printf("BC                %+25.15Le []\n",BC); 
 printf("\n");

  long double htHUH=c*sqrt(Efac*Efac-kapSqu);
//long double htHUH=c*sqrt(1.L/Efac/Efac-kapSqu);
 printf("hthetaOf0################################################################\n"); 
 printf("htHUH             %+25.15Le [m/s]   <<<---   ???   Munoz (32)   ???\n",htHUH); 
 printf("Efac*Efac         %+25.15Le []\n",Efac*Efac); 
 printf("kapSqu            %+25.15Le []\n",kapSqu); 
 printf("Efac*Efac-kapSqu  %+25.15Le []\n",Efac*Efac-kapSqu); 
 printf("\n");

 long double htCHK=kapSqu*L0/mp/r0-kD*EM0/L0/mp/c/c;
 printf("htCHK             %+25.15Le [m/s]                  Munoz (16)\n",htCHK); 
 long double htCHK2=L0/mp/r0-kD*EM0/L/mp/c/c-kD*kD/r0/L0/mp/c/c;
 printf("htCHK2            %+25.15Le [m/s]                  Munoz (15)\n",htCHK2); 
 printf("A                 %+25.15Le [m/s]                  ME\n",A); 
 printf("hthetaOf0################################################################\n"); 
 printf("\n");
 long double roCHK=lambda/(1.L+AC);
 printf("roCHK             %+25.15Le [m]\n",roCHK); 
 long double roCHK2=lambda/(1.L-AC);
 printf("roCHK2            %+25.15Le [m]\n",roCHK2); 
 printf("\n");

 long double thDD=sqrt(kD/gD/mp/rD/rD/rD);
 printf("thDD              %+25.15Le [1/s]\n",thDD); 
 printf("\n");

 long double ht=0.;

 long double Ngrid=atof(argv[9]);
 long double delThta=2.*M_PI/Ngrid;
 long double delThtH=delThta/2.L;
 long double tau=2.*M_PI;
 long double Thta=0.; 
 long double ThtM=0.; 
// long double r=0.; 
   long double rL=r0; 
   long double rA=0.; 
 long double x=0.; 
 long double y=0.; 
 long double dt=0.;
 long double T=0.;
 long double TD=2.*M_PI*rD/vD;

 std::ofstream fp("2D_5Values.out");
 for(Thta=0.;Thta<tau;Thta+=delThta){
  ThtM=Thta+delThtH;
//printf("ThtM              %+25.15Le [s]\n",ThtM); 
  r=lambda/( 1.L+AC*cos(kap*ThtM)+BC*sin(kap*ThtM) );
  x=r*cos(Thta);
  y=r*sin(Thta);
//dt=r/k*(L-mp*r*ht)*delThta;
  dt=r/L/c/c*(kD+r*EM)*delThta;
  T=T+dt;
  fp << x << " " << y << "\n";
 }

 printf("Orbital Period***********************************************************\n"); 
 printf("AC (should be << 1) %+25.15Le [s]\n",AC); 
 printf("BC (should be << 1) %+25.15Le [s]\n",BC); 
 printf("TD (Design)         %+25.15Le [s]\n",TD); 
 printf("T (simple sum)      %+25.15Le [s]\n",T); 
 printf("Orbital Period***********************************************************\n"); 
 printf("\n");

 long double Escr=TE_cnv*GeVperJ;
 long double kRTFac=dr0/rD/(rD+dr0);

 printf("Theoretical Delta Values (Input Radius Treated As Circle)\n");
 printf("______________________________________________________________________________________\n");
 printf("L=L0, cnsrvd ang mmntm                       %+25.15Le [kg*m*m/s]\n",L0); 
 printf("RE_Mz Rest Energy Munoz                      %+25.15Le [J]\n",RE_Mz); 
 printf("KE_Mz=KE_Mz0 Kinetic Energy Munoz            %+25.15Le [J]\n",KE_Mz0); 
 printf("PE_Mz=PE_Mz0 Potential Energy Munoz          %+25.15Le [J]\n",PE_Mz0); 
 printf("RE_Mz+KE_Mz+PE_Mz Total Energy Munoz         %+25.15Le [J]\n",RE_Mz+KE_Mz+PE_Mz); 
 printf("TE_Mz Total Energy Munoz                     %+25.15Le [J]\n",TE_Mz); 
 printf("TE_Mz*GeVperJ                                %+25.15Le [GeV]\n",TE_Mz*GeVperJ); 
 printf("TE_Mz*GeVperJ*TheorGDelta                    %+25.15Le [GeV]\n",TE_Mz*GeVperJ*TheorGDelta); 
 printf("\n");
 printf("\033[1m\033[31m");
// printf("\033[1m\033[31m" "\n");
 printf("RE_Mz*GeVperJ       Rest Energy Conventional %+25.15Le [GeV]\n",RE_Mz*GeVperJ); 
 printf("KE_Mz*GeVperJ    Kinetic Energy Conventional %+25.15Le [GeV]\n",KE_Mz0*GeVperJ); 
 printf(RESET);
 printf("\n");
 printf("kD*GeVperJ/rD                                %+25.15Le [GeV]\n",kD*GeVperJ/rD); 
 printf("PE_Mz*GeVperJ                                %+25.15Le [GeV]\n",PE_Mz0*GeVperJ); 
 printf("\033[1m\033[31m");
 printf("PE_cnv*GeVperJ Potential Energy Conventional %+25.15Le [GeV]\n",PE_cnv*GeVperJ); 
 printf(RESET);
 printf("(-kD/rDelta+kD/rD)*GeVperJ                   %+25.15Le [GeV]\n",(-kD/rDelta+kD/rD)*GeVperJ); 
 printf(RESET);
 printf("\n");
 printf("(RE_Mz+KE_Mz+PE_Mz)*GeVperJ                  %+25.15Le [GeV]\n",(RE_Mz+KE_Mz+PE_Mz)*GeVperJ); 
 printf("(RE_Mz+KE_Mz+PE_cnv)*GeVperJ                 %+25.15Le [GeV]\n",(RE_Mz+KE_Mz+PE_cnv)*GeVperJ); 
 printf("(RE_Mz+KE_Mz+PE_Mz+kRT/rD)*GeVperJ           %+25.15Le [GeV]\n",(RE_Mz+KE_Mz+PE_Mz+kRT/rD)*GeVperJ); 
 printf("\033[1m\033[31m");
 printf("TE_cnv*GeVperJ     Total Energy Conventional %+25.15Le [GeV]\n",TE_cnv*GeVperJ); 
 printf(RESET);
 printf("\n");
 printf("Escr                                         %+25.15Le [GeV]\n",Escr); 
 printf("______________________________________________________________________________________\n");
 printf("\n");

 printf("Theoretical Delta Quantities \n");
 printf("______________________________________________________________________________________\n");
 printf("r=r0=rDelta, initial radius                  %+25.15Le [m]\n",rDelta); 
 printf("rD, design radius                            %+25.15Le [m]\n",rD); 
 printf("L=L0, cnsrvd ang mmntm                       %+25.15Le [kg*m*m/s]\n",L0); 
 printf("TheorTDDelta                                 %+25.15Le []\n",TheorTDDelta); 
 printf("TheorVDelta                                  %+25.15Le [m/s]\n",TheorVDelta); 
 printf("TheorBDelta                                  %+25.15Le []\n",TheorBDelta); 
 printf("TheorGDelta                                  %+25.15Le []\n",TheorGDelta); 
 printf("gFS                                          %+25.15Le []\n",gFS); 
 printf("\n");
 printf("TheorPDelta                                  %+25.15Le [kg m / s]\n",TheorPDelta); 
 printf("TheorPDeltac                                 %+25.15Le [J]\n",TheorPDeltac); 
 printf("TheorPDeltacEV                               %+25.15Le [GeV]\n",TheorPDeltacEV); 
 printf("TheorMEDeltaEV                               %+25.15Le [GeV]\n",TheorMEDeltaEV); 
 printf("TheorPEDeltaEV                               %+25.15Le [GeV]\n",TheorPEDeltaEV); 
 printf("TheorTEDeltaEV                               %+25.15Le [GeV]\n",TheorTEDeltaEV); 
 printf("ERT                                          %+25.15Le [GeV]\n",ERT); 
 printf("______________________________________________________________________________________\n");
 printf("\n");

 printf("./2D_5Values kD(Design) rD(esign) gD(esign) dr0 th0 thD0 rD0 dg0 Ngrid\n"); 
 printf("./2D_5Values %+16.10Le %+16.10Le %+16.10Le %+16.10Le %+16.10Le %+16.10Le %+16.10Le %+16.10Le %+16.10Le \n",kD,rD,gD,dr0,th0,thD0,rD0,dg0,Ngrid); 
 printf("\n");
 printf("TheorTEDeltaEV - ERT                         %+25.15Le [GeV]\n",TheorTEDeltaEV-ERT); 

/*
 printf("\nTheoretical Design Quantities \n");
 printf("_________________________________\n");
 printf("TheorTDDelta                %+25.15Le []\n",TheorTDDelta); 
 printf("TheorVDelta                 %+25.15Le [m/s]\n",TheorVDelta); 
 printf("TheorBDelta                 %+25.15Le []\n",TheorBDelta); 
 printf("TheorGDelta                 %+25.15Le []\n",TheorGDelta); 
 printf("gFS                         %+25.15Le []\n",gFS); 
 printf("TheorPDelta                 %+25.15Le [kg m / s]\n",TheorPDelta); 
 printf("TheorPDeltac                %+25.15Le [J]\n",TheorPDeltac); 
 printf("TheorPDeltacEV              %+25.15Le [GeV]\n",TheorPDeltacEV); 
 printf("ElD, elc fld dsgn           %+25.15Le [V/m]\n",ElD);
 printf("ElRT                        %+25.15Le [V/m]\n",ElRT);
 printf("kRT                         %+25.15Le [J*m]\n",kRT);
 printf("kMCnstrn                    %+25.15Le [J*m] - this should equal k Munoz\n",kMCnstrn); 
 printf("PE_cnv0*GeVperJ                 %+25.15Le [GeV]\n",PE_cnv0*GeVperJ); 
 printf("TE_cnv*GeVperJ                 %+25.15Le [GeV]\n",TE_cnv*GeVperJ); 
 printf("Escr                        %+25.15Le [GeV]\n",Escr); 
 printf("_________________________________\n");
 printf("\n");
*/

}
