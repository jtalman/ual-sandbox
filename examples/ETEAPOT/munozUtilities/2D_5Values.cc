#include<iostream>
#include<fstream>

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"../clr"
#include"../../../codes/UAL/src/UAL/Common/Def.hh"
//#include"Def.hh"

int main(int argc,char*argv[]){
 long double mpG=UAL::pmass;
 long double ep=UAL::elemCharge;
 long double c=UAL::clight;
 long double vP=UAL::vcmPerm;
 long double kC=1./4./UAL::pi/vP;

 if(argc!=10){
  printf("Usage: ./2D_5Values k(Design) rD(esign) gD(esign) dr th0 thD0 rD0 dg0 Ngrid\n");
  exit(1);
 }

 long double mp=mpG*ep*1.e+9/c/c;
 long double Ep=mp*c*c;

// system constants
 long double k=atof(argv[1]);
 long double rD=atof(argv[2]); 
 long double gD=atof(argv[3]); 

 long double bD=sqrt(1.-1./gD/gD);
 long double vD=bD*c;
 long double pD=gD*mp*vD;
 long double LD=rD*pD;

 long double LC=k/c;
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
 printf("k, Munoz          %+25.15Le [J*m]\n",k);
 printf("rD, rds dsgn      %+25.15Le [m]\n",rD);
 printf("gD, gamma dsgn    %+25.15Le []\n",gD);
 printf("\n");
 printf("bD, beta dsgn     %+25.15Le []\n",bD);
 printf("vD, vlcty dsgn    %+25.15Le [m/s]\n",vD);
 printf("pD, momen dsgn    %+25.15Le [kg*m/s]\n",pD);
 printf("LD, ang mom dsgn  %+25.15Le [kg*m*m/s]\n",LD);
 printf("LC, ang mom crtcl %+25.15Le [kg*m*m/s] - k/c\n",LC); 
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

 printf("\nTheoretical Design Quantities \n");
 printf("_________________________________\n");
 long double aQ         = 1.;
 long double bQ         = (k/mp/rD/rD/c)*(k/mp/rD/rD/c);
 long double cQ         = -(k/mp/rD/rD/rD)*(k/mp/rD/rD/rD);
 long double TheorTDD_sq= (-bQ + sqrt(bQ*bQ - 4.*aQ*cQ))/2./aQ;
 long double TheorTDD   = sqrt(TheorTDD_sq);
 long double TheorVD    = rD*TheorTDD;
 long double TheorGD    = 1./sqrt(1.-TheorVD*TheorVD/c/c);
 long double TheorPD    = TheorGD * mp * TheorVD;
 long double TheorPDc   = TheorPD*c;
 long double q=ep; 
 long double GeVperJ=1./q/1.e9;
 long double TheorPDcEV = TheorPDc*GeVperJ;
 printf("TheorTDD          %+25.15Le []\n",TheorTDD); 
 printf("TheorVD           %+25.15Le []\n",TheorVD); 
 printf("TheorGD           %+25.15Le []\n",TheorGD); 
 printf("TheorPD           %+25.15Le [kg m / s]\n",TheorPD); 
 printf("TheorPDc          %+25.15Le [J]\n",TheorPDc); 
 printf("TheorPDcEV        %+25.15Le [GeV]\n",TheorPDcEV); 
 printf("_________________________________\n");
 printf("\n");

 printf("\nConstraints\n");
 printf("_________________________________\n");
 long double ElD=k/ep/rD/rD;
 long double kMCnstrn=mp*rD*(gD-1./gD)*c*c;
 long double QM=-ElD*rD*rD/kC; 
 long double kMCnstrn2=kC*QM*ep;
// long double q=ep; 
 printf("ElD, elc fld dsgn %+25.15Le [V/m]\n",ElD);
 printf("kMCnstrn          %+25.15Le [J*m] - this should equal k Munoz\n",kMCnstrn); 
 printf("QM, fixed charge  %+25.15Le [C]\n",QM); 
 printf("kC QM q/rD/rD     %+25.15Le [Nt] - should equal below\n",kC*QM*q/rD/rD); 
 printf("q ElD             %+25.15Le [Nt] - should equal above\n",q*ElD); 
 printf("kMCnstrn2         %+25.15Le [Jm] - should equal argv[1]\n",kMCnstrn2); 
 printf("_________________________________\n");
 printf("\n");

 long double g0=gD+dg0;
 long double g=g0;
 long double r0=rD+dr0;
 long double r=r0;

 long double L0=g0*mp*r0*r0*thD0;
 long double L=L0;

 long double MEM0=g*mp*c*c;
 long double MEM=MEM0;
 long double PEM0=-k/r;
 long double PEM=PEM0;
 long double EM0=MEM0+PEM0;
 long double EC0=MEM0+PEM0+k/rD;
// long double GeVperJ=1./q/1.e9;
 long double EM=EM0;
 printf("\nDynamic Constants (Conserved Quantities and Their Components)\n");
 printf("_________________________________\n");
 printf("r,  initial radius%+25.15Le [m]\n",r); 
 printf("r0, initial radius%+25.15Le [m]\n",r0); 
 printf("rD, design  radius%+25.15Le [m]\n",rD); 
 printf("L=L0, cnsrv ng mm %+25.15Le [kg*m*m/s]\n",L0); 
 printf("\n");
 printf("MEM=MEM0,mech ngy %+25.15Le [J]\n",MEM0); 
 printf("PEM=PEM0,pot  ngy %+25.15Le [J]\n",PEM0); 
 printf("\n");
 printf("EM=EM0, cnsrv ngy %+25.15Le [J]\n",EM0); 
 printf("EC=EC0, cnsrv ngy %+25.15Le [J]\n",EC0); 
 printf("GeVperJ           %+25.15Le [J]\n",GeVperJ); 
 printf("EC0*GeVperJ       %+25.15Le [GeV]\n",EC0*GeVperJ); 
 printf("Ep/gD             %+25.15Le [J]\n",Ep/gD); 
 printf("_________________________________\n");
 printf("\n");

 printf("\nStarting Dynamic Variables\n");
 printf("_________________________________\n");
 printf("g, gamma          %+25.15Le [] - \"g0\"\n",g0); 
 printf("r0, radius        %+25.15Le [m] - \"r0\"\n",r0); 

 long double LcOver_k=L*c/k;
 long double Efac=EM/Ep;
 long double kapSqu  = 1.-1./LcOver_k/LcOver_k;
 long double kap=sqrt(kapSqu);
 long double lambda=kapSqu*L0*L0*c*c/k/EM;   //   40;
 long double eps=0.;
 if(1.-kapSqu/Efac/Efac>0){
  eps=LcOver_k*sqrt(1.-kapSqu/Efac/Efac);
 }
 else{
//printf("\"d\" 1.-kapSqu/Efac/Efac %+25.15Le\n",1.-kapSqu/Efac/Efac);
 }
// long double r0=lambda/(1.+eps);                                 <<<---   implement consistency check?
 printf("\n");
 printf("v0 (r0*thD0)      %+25.15Le [m/s]\n",r0*thD0); 
 printf("_________________________________\n");
 printf("\n");

 long double b=1./Efac/Efac-1.;
 long double a = 1./LcOver_k;
 long double eps2=0.L;
 if(a*a-b+b*a*a > 0){
  eps2=1.L/a*sqrt( a*a - b + b*a*a );
 }
 else{
//printf("\"d\" a*a - b + b*a*a %+25.15Le\n",a*a - b + b*a*a);
 }
 long double a2=b/sqrt(1.+b);

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
 printf("eps               %+25.15Le [] (\"d\" 1.-kapSqu/Efac/Efac %+25.15Le)\n",eps,1.-kapSqu/Efac/Efac); 
 printf("eps2              %+25.15Le [] (\"d\" a*a - b + b*a*a     %+25.15Le)\n",eps2,a*a - b + b*a*a); 
 printf("_________________________________\n");
 printf("*********************************\n");
 printf("\n");
 printf(RESET);

 long double A=g0*r0*thD0-k*g0/L0;
 printf("A                 %+25.15Le [m/s]\n",A); 
 long double C=L0*mp*c*c/k/EM0;
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
//long double htHUH=c*sqrt(1./Efac/Efac-kapSqu);
 printf("hthetaOf0################################################################\n"); 
 printf("htHUH             %+25.15Le [m/s]   <<<---   ???   Munoz (32)   ???\n",htHUH); 
 printf("Efac*Efac         %+25.15Le []\n",Efac*Efac); 
 printf("kapSqu            %+25.15Le []\n",kapSqu); 
 printf("Efac*Efac-kapSqu  %+25.15Le []\n",Efac*Efac-kapSqu); 
 printf("\n");

 long double htCHK=kapSqu*L0/mp/r0-k*EM0/L0/mp/c/c;
 printf("htCHK             %+25.15Le [m/s]                  Munoz (16)\n",htCHK); 
 long double htCHK2=L0/mp/r0-k*EM0/L/mp/c/c-k*k/r0/L0/mp/c/c;
 printf("htCHK2            %+25.15Le [m/s]                  Munoz (15)\n",htCHK2); 
 printf("A                 %+25.15Le [m/s]                  ME\n",A); 
 printf("hthetaOf0################################################################\n"); 
 printf("\n");
 long double roCHK=lambda/(1.+AC);
 printf("roCHK             %+25.15Le [m]\n",roCHK); 
 long double roCHK2=lambda/(1.-AC);
 printf("roCHK2            %+25.15Le [m]\n",roCHK2); 
 printf("\n");

 long double thDD=sqrt(k/gD/mp/rD/rD/rD);
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
  r=lambda/( 1.+AC*cos(kap*ThtM)+BC*sin(kap*ThtM) );
  x=r*cos(Thta);
  y=r*sin(Thta);
//dt=r/k*(L-mp*r*ht)*delThta;
  dt=r/L/c/c*(k+r*EM)*delThta;
  T=T+dt;
  fp << x << " " << y << "\n";
 }

 printf("Orbital Period***********************************************************\n"); 
 printf("TD (Design)       %+25.15Le [s]\n",TD); 
 printf("T (simple sum)    %+25.15Le [s]\n",T); 
 printf("Orbital Period***********************************************************\n"); 
 printf("\n");

}
