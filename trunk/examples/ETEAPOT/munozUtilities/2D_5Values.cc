#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"../clr"
#include"../../../codes/UAL/src/UAL/Common/Def.hh"
//#include"Def.hh"

int main(int argc,char*argv[]){
 double mpG=UAL::pmass;
 double ep=UAL::elemCharge;
 double c=UAL::clight;
 double vP=UAL::vcmPerm;
 double kC=1./4./UAL::pi/vP;

 if(argc!=9){
  printf("Usage: ./2D_5Values k(Munoz) rD(esign) gD(esign) dr th0 thD0 rD0 dg0\n");
  exit(1);
 }

 double mp=mpG*ep*1.e+9/c/c;
 double Ep=mp*c*c;

// system constants
 double k=atof(argv[1]);
 double rD=atof(argv[2]); 
 double gD=atof(argv[3]); 

 double bD=sqrt(1.-1./gD/gD);
 double vD=bD*c;
 double pD=gD*mp*vD;
 double LD=rD*pD;

 double LC=k/c;
// system constants

// initial variables
 double dr0=atof(argv[4]);
 double dx0=dr0;
 double x0=rD+dr0;
 double th0=atof(argv[5]);
 double thD0=atof(argv[6]);
 double rD0=atof(argv[7]);
 double dg0=atof(argv[8]);
// initial variables

 printf(RESET);
 printf("\nPhysical Constants\n");
 printf("_________________________________\n");
 printf("mp, proton mass   %+20.10e [kg]\n",mp);
 printf("Ep (mp * c * c)   %+20.10e [J]\n",Ep);
 printf("ep, proton chge   %+20.10e [C] - also designated q below\n",ep);
 printf("c, speed light    %+20.10e [m/s]\n",c);
 printf("vP, vacuum perm   %+20.10e [F/m]\n",vP);
 printf("kC, Coulomb       %+20.10e [m/F]\n",kC);
 printf("_________________________________\n");
 printf("\n");

 printf("\nSystem Constants - D for Design\n");
 printf("_________________________________\n");
 printf("k, Munoz          %+20.10e [J*m]\n",k);
 printf("rD, rds dsgn      %+20.10e [m]\n",rD);
 printf("gD, gamma dsgn    %+20.10e []\n",gD);
 printf("\n");
 printf("bD, beta dsgn     %+20.10e []\n",bD);
 printf("vD, vlcty dsgn    %+20.10e [m/s]\n",vD);
 printf("pD, momen dsgn    %+20.10e [kg*m/s]\n",pD);
 printf("LD, ang mom dsgn  %+20.10e [kg*m*m/s]\n",LD);
 printf("LC, ang mom crtcl %+20.10e [kg*m*m/s] - k/c\n",LC); 
 printf("LD/LC             %+20.10e []\n",LD/LC); 
 printf("_________________________________\n");
 printf("\n");

 printf("\nInitial Probe Parameter Values \"...0\"\n");
 printf("_________________________________\n");
 printf("dr0               %+20.10e [m]\n",dr0);
 printf("dx0               %+20.10e [m]\n",dx0);
 printf("x0                %+20.10e [m]\n",x0);
 printf("th0               %+20.10e []\n",th0);
 printf("thD0              %+20.10e [1/s]\n",thD0);
 printf("rD0               %+20.10e [m/s]\n",rD0);
 printf("dg0               %+20.10e []\n",dg0); 
 printf("_________________________________\n");
 printf("\n");

 printf("\nConstraints\n");
 printf("_________________________________\n");
 double ElD=k/ep/rD/rD;
 double kMCnstrn=mp*rD*(gD-1./gD)*c*c;
 double QM=-ElD*rD*rD/kC; 
 double kMCnstrn2=kC*QM*ep;
 double q=ep; 
 printf("ElD, elc fld dsgn %+20.10e [V/m]\n",ElD);
 printf("kMCnstrn          %+20.10e [J*m] - this should equal k Munoz\n",kMCnstrn); 
 printf("QM, fixed charge  %+20.10e [C]\n",QM); 
 printf("kC QM q/rD/rD     %+20.10e [Nt] - should equal below\n",kC*QM*q/rD/rD); 
 printf("q ElD             %+20.10e [Nt] - should equal above\n",q*ElD); 
 printf("kMCnstrn2         %+20.10e [Jm] - should equal argv[1]\n",kMCnstrn2); 
 printf("_________________________________\n");
 printf("\n");

 double g0=gD+dg0;
 double g=g0;
 double r0=rD+dr0;
 double r=r0;

 double L0=g0*mp*r0*r0*thD0;
 double L=L0;

 double MEM0=g*mp*c*c;
 double PEM0=-k/r;
 double EM0=g*mp*c*c-k/r;
 double EM=EM0;
 printf("\nDynamic Constants (Conserved Quantities and Their Components)\n");
 printf("_________________________________\n");
 printf("L=L0, cnsrv ng mm %+20.10e [kg*m*m/s]\n",L0); 
 printf("\n");
 printf("MEM=MEM0,mech ngy %+20.10e [J]\n",MEM0); 
 printf("PEM=PEM0,pot  ngy %+20.10e [J]\n",PEM0); 
 printf("\n");
 printf("EM=EM0, cnsrv ngy %+20.10e [J]\n",EM0); 
 printf("Ep/gD             %+20.10e [J]\n",Ep/gD); 
 printf("_________________________________\n");
 printf("\n");

 printf("\nStarting Dynamic Variables\n");
 printf("_________________________________\n");
 printf("g, gamma          %+20.10e [] - \"g0\"\n",g0); 
 printf("r, radius         %+20.10e [m] - \"r0\"\n",r0); 
 printf("_________________________________\n");
 printf("\n");

 double lambda=40;
 double LcOver_k=L*c/k;
 double Efac=EM/Ep;
 double kapSqu  = 1.-1./LcOver_k/LcOver_k;
 double kap=sqrt(kapSqu);
 double eps=0.;
 if(1.-kapSqu/Efac/Efac>0){
  eps=LcOver_k*sqrt(1.-kapSqu/Efac/Efac);
 }
 else{
//printf("\"d\" 1.-kapSqu/Efac/Efac %+20.10e\n",1.-kapSqu/Efac/Efac);
 }
 double R0=lambda/(1.+eps);

 double b=1./Efac/Efac-1.;
 double a = 1./LcOver_k;
 double eps2=0.L;
 if(a*a-b+b*a*a > 0){
  eps2=1.L/a*sqrt( a*a - b + b*a*a );
 }
 else{
//printf("\"d\" a*a - b + b*a*a %+20.10e\n",a*a - b + b*a*a);
 }
 double a2=b/sqrt(1.+b);

  printf("\n" "\033[1m\033[31m" "PURE MUNOZ CROSS CHECK\n");
//printf("\n" "\033[1m\033[30m" "PURE MUNOZ CROSS CHECK\n");
 printf("_________________________________\n");
 printf("*********************************\n");
 printf("lambda            %+20.10e [m] - just for completeness, lambda plays no role here\n",lambda); 
 printf("LcOver_k          %+20.10e []\n",LcOver_k); 
 printf("Efac              %+20.10e []\n",Efac); 
 printf("eps               %+20.10e [] (\"d\" 1.-kapSqu/Efac/Efac %+20.10e)\n",eps,1.-kapSqu/Efac/Efac); 
 printf("eps2              %+20.10e [] (\"d\" a*a - b + b*a*a %+20.10e)\n",eps2,a*a - b + b*a*a); 
 printf("_________________________________\n");
 printf("*********************************\n");
 printf("\n");
 printf(RESET);

}