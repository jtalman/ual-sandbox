#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iomanip>
#include "PAC/Beam/Position.hh"

  #define sElems 366

using namespace std;
int main(int argc, char* argv[]){
 int ip=1;
 int bend=0;
 string alp;
 int i;
 PAC::Position p1[sElems],p2[sElems],p3[sElems],p4[sElems],p5[sElems],p6[sElems],p7[sElems],p8[sElems],p9[sElems],p10[sElems];
 double Mtld[sElems][6][6];

 cout << setiosflags( ios::showpos    );  
 cout << setiosflags( ios::uppercase  );  
 cout << setiosflags( ios::scientific );
 cout << setfill( ' ' );
 cout << setiosflags( ios::left );
 cout << setprecision(13) ;

 char bfr[10];
 sprintf(bfr,"%+5.2f",atof(argv[1]));
 string m(bfr);
 cerr << "m          " << m          << "\n";
 cerr << "m.length() " << m.length() << "\n";

#include "sip1"
#include "sip2"
#include "sip3"
#include "sip4"
#include "sip5"
#include "sip6"
#include "sip7"
#include "sip8"
#include "sip9"
#include "sip10"

 double PI=3.141592654;

 double a0x=atof(argv[2]);
 double b0x=atof(argv[3]);
 double muX_OfS;
 double xTrace;
 double xMu;
 double psiX_OfS;
 double betaX_OfS;

 double a0y=atof(argv[4]);
 double b0y=atof(argv[5]);
 double muY_OfS;
 double yTrace;
 double yMu;
 double psiY_OfS;
 double betaY_OfS;

// i=atoi(argv[2]);
 for(int i=0;i<sElems;i++){
#include "probeDataForTwiss"

//#include "col0"
  Mtld[i][0][0]=(p1[i][0]-p2[i][0])/2/x1typ;
  Mtld[i][1][0]=(p1[i][1]-p2[i][1])/2/x1typ;
  Mtld[i][2][0]=(p1[i][2]-p2[i][2])/2/x1typ;
  Mtld[i][3][0]=(p1[i][3]-p2[i][3])/2/x1typ;
  Mtld[i][4][0]=(p1[i][4]-p2[i][4])/2/x1typ;
  Mtld[i][5][0]=(p1[i][5]-p2[i][5])/2/x1typ;

#include "col1tld"
#include "col2tld"
#include "col3tld"
#include "col4tld"
#include "col5tld"

  muX_OfS=atan2( Mtld[i][0][1],b0x*Mtld[i][0][0]-a0x*Mtld[i][0][1] );
//cout << "   muX_OfS              " <<    muX_OfS             << "\n\n";
  xTrace=Mtld[i][0][0]+Mtld[i][1][1];
  xMu=acos(xTrace/2);
//cout << "  acos(xTrace/2)      " << acos(xTrace/2)        << "\n";
//cout << "  acos(xTrace/2)/2/PI " << acos(xTrace/2)/2/PI   << "\n";
//cout << "1+acos(xTrace/2)/2/PI " << 1+acos(xTrace/2)/2/PI << "\n\n";
  psiX_OfS=muX_OfS;
  betaX_OfS=Mtld[i][0][1]*Mtld[i][0][1]/sin(psiX_OfS)/sin(psiX_OfS)/b0x;

  muY_OfS=atan2( Mtld[i][2][3],b0y*Mtld[i][2][2]-a0y*Mtld[i][2][3] );
  yTrace=Mtld[i][2][2]+Mtld[i][3][3];
  yMu=acos(yTrace/2);
  psiY_OfS=muY_OfS;
  betaY_OfS=Mtld[i][2][3]*Mtld[i][2][3]/sin(psiY_OfS)/sin(psiY_OfS)/b0y;

  cout << i << " " << betaX_OfS << " " << betaY_OfS << "\n";
 }
 return 0;
}
