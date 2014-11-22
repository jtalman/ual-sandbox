#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iomanip>
#include "PAC/Beam/Position.hh"

//#define sElems 1006
//#define sElems 286
//#define sElems 206
//#define sElems 160

using namespace std;
int main(int argc, char* argv[]){

 if( argc != 7 ){
  cerr << "Usage: ./transferMatrices +1.0 0.0 31.9793 0.0 77.1718 206 > ! betaFunctions\n";
  exit(1);
 }

 int sElems = atoi( argv[6] );
 int ip=1;
 int bend=0;
 string alp;
 int i;
 PAC::Position p1[sElems],p2[sElems],p3[sElems],p4[sElems],p5[sElems],p6[sElems],p7[sElems],p8[sElems],p9[sElems],p10[sElems];
 double M[sElems][6][6];

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
 double psiX_OfS=0.0;
 double betaX_OfS;

 double a0y=atof(argv[4]);
 double b0y=atof(argv[5]);
 double muY_OfS;
 double yTrace;
 double yMu;
 double psiY_OfS=0.0;
 double betaY_OfS;

 double tolerance=+0.000001;
 double betaX_OfSLAST=b0x;
 double betaY_OfSLAST=b0y;
 double psiX_OfSLAST=0.0;
 double psiY_OfSLAST=0.0;
 int intTuneX_LAST=0;
 int intTuneY_LAST=0;

#include "probeDataForTwiss"
// i=atoi(argv[2]);
 for(int i=0;i<sElems;i++){

  // #include "col0"
  M[i][0][0]=(p1[i][0]-p2[i][0])/2/x1typ;
  M[i][1][0]=(p1[i][1]-p2[i][1])/2/x1typ;
  M[i][2][0]=(p1[i][2]-p2[i][2])/2/x1typ;
  M[i][3][0]=(p1[i][3]-p2[i][3])/2/x1typ;
  M[i][4][0]=(p1[i][4]-p2[i][4])/2/x1typ;
  M[i][5][0]=(p1[i][5]-p2[i][5])/2/x1typ;

  // #include "col1tld"
  M[i][0][1]=(p3[i][0]-p4[i][0])/2/x2typ;
  M[i][1][1]=(p3[i][1]-p4[i][1])/2/x2typ;
  M[i][2][1]=(p3[i][2]-p4[i][2])/2/x2typ;
  M[i][3][1]=(p3[i][3]-p4[i][3])/2/x2typ;
  M[i][4][1]=(p3[i][4]-p4[i][4])/2/x2typ;
  M[i][5][1]=(p3[i][5]-p4[i][5])/2/x2typ;

  // #include "col2tld"
  M[i][0][2]=(p5[i][0]-p6[i][0])/2/y1typ;
  M[i][1][2]=(p5[i][1]-p6[i][1])/2/y1typ;
  M[i][2][2]=(p5[i][2]-p6[i][2])/2/y1typ;
  M[i][3][2]=(p5[i][3]-p6[i][3])/2/y1typ;
  M[i][4][2]=(p5[i][4]-p6[i][4])/2/y1typ;
  M[i][5][2]=(p5[i][5]-p6[i][5])/2/y1typ;

  // #include "col3tld"
  M[i][0][3]=(p7[i][0]-p8[i][0])/2/y2typ;
  M[i][1][3]=(p7[i][1]-p8[i][1])/2/y2typ;
  M[i][2][3]=(p7[i][2]-p8[i][2])/2/y2typ;
  M[i][3][3]=(p7[i][3]-p8[i][3])/2/y2typ;
  M[i][4][3]=(p7[i][4]-p8[i][4])/2/y2typ;
  M[i][5][3]=(p7[i][5]-p8[i][5])/2/y2typ;

  // #include "col4tld"
  M[i][0][4]=0;
  M[i][1][4]=0;
  M[i][2][4]=0;
  M[i][3][4]=0;
  M[i][4][4]=1;
  M[i][5][4]=0;

  // #include "col5tld"
  M[i][0][5]=(p9[i][0]-p10[i][0])/2/deltyp;
  M[i][1][5]=(p9[i][1]-p10[i][1])/2/deltyp;
  M[i][2][5]=(p9[i][2]-p10[i][2])/2/deltyp;
  M[i][3][5]=(p9[i][3]-p10[i][3])/2/deltyp;
  M[i][4][5]=(p9[i][4]-p10[i][4])/2/deltyp;
  M[i][5][5]=(p9[i][5]-p10[i][5])/2/deltyp;

  psiX_OfS=2*PI*intTuneX_LAST + atan2( M[i][0][1], b0x*M[i][0][0]-a0x*M[i][0][1] );
  if( psiX_OfS < psiX_OfSLAST ){ 
    intTuneX_LAST++;
    psiX_OfS+=2*PI;
  }
  psiX_OfSLAST = psiX_OfS;

  if( fabs( sin(psiX_OfS) ) > tolerance ){
    betaX_OfS = M[i][0][1]*M[i][0][1]/sin(psiX_OfS)/sin(psiX_OfS)/b0x;
    betaX_OfSLAST = betaX_OfS;
  }
  else{
   betaX_OfS = betaX_OfSLAST;
  }
  betaX_OfSLAST = betaX_OfS;

  psiY_OfS=2*PI*intTuneY_LAST + atan2( M[i][2][3],b0y*M[i][2][2]-a0y*M[i][2][3] );
  if( psiY_OfS < psiY_OfSLAST ){ 
    intTuneY_LAST++;
    psiY_OfS+=2*PI;
  }
  psiY_OfSLAST = psiY_OfS;

  if( fabs( sin(psiY_OfS) ) > tolerance ){
    betaY_OfS = M[i][2][3]*M[i][2][3]/sin(psiY_OfS)/sin(psiY_OfS)/b0y;
    betaY_OfSLAST = betaY_OfS;
  }
  else{
   betaY_OfS = betaY_OfSLAST;
  } 
  betaY_OfSLAST = betaY_OfS;

  cout << setw(12) << name[i] << " " << s[i] << " " << betaX_OfSLAST << " " << betaY_OfSLAST << " " << psiX_OfSLAST << " " <<  psiY_OfSLAST << "\n";
 }
 return 0;
}
