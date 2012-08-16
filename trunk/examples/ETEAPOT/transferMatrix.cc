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
 double Mi[6][6];

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

 i=atoi(argv[2]);
#include "probeDataForTwiss"

//#include "col0"
 Mi[0][0]=(p1[i][0]-p2[i][0])/2/x1typ;
 Mi[1][0]=(p1[i][1]-p2[i][1])/2/x1typ;
 Mi[2][0]=(p1[i][2]-p2[i][2])/2/x1typ;
 Mi[3][0]=(p1[i][3]-p2[i][3])/2/x1typ;
 Mi[4][0]=(p1[i][4]-p2[i][4])/2/x1typ;
 Mi[5][0]=(p1[i][5]-p2[i][5])/2/x1typ;
#include "col1"
#include "col2"
#include "col3"
#include "col4"
#include "col5"

 cout << Mi[0][0] << " " << Mi[0][1] << " " << Mi[0][2] << " " << Mi[0][3] << " " << Mi[0][4] << " " << Mi[0][5] << "\n";
 cout << Mi[1][0] << " " << Mi[1][1] << " " << Mi[1][2] << " " << Mi[1][3] << " " << Mi[1][4] << " " << Mi[1][5] << "\n";
 cout << Mi[2][0] << " " << Mi[2][1] << " " << Mi[2][2] << " " << Mi[2][3] << " " << Mi[2][4] << " " << Mi[2][5] << "\n";
 cout << Mi[3][0] << " " << Mi[3][1] << " " << Mi[3][2] << " " << Mi[3][3] << " " << Mi[3][4] << " " << Mi[3][5] << "\n";
 cout << Mi[4][0] << " " << Mi[4][1] << " " << Mi[4][2] << " " << Mi[4][3] << " " << Mi[4][4] << " " << Mi[4][5] << "\n";
 cout << Mi[5][0] << " " << Mi[5][1] << " " << Mi[5][2] << " " << Mi[5][3] << " " << Mi[5][4] << " " << Mi[5][5] << "\n";

 double a0x=atof(argv[3]);
 double b0x=atof(argv[4]);
 double tanMuOfS=Mi[0][1]/(b0x*Mi[0][0]-a0x*Mi[0][1]);
  double    muOfS=atan2( Mi[0][1],b0x*Mi[0][0]-a0x*Mi[0][1] );
//double    muOfS=atan ( tanMuOfS );
 cout << "tanMuOfS              " << tanMuOfS             << "\n";
 cout << "   muOfS              " <<    muOfS             << "\n\n";

 double PI=3.141592654;
 double xTrace;
        xTrace=Mi[0][0]+Mi[1][1];
 double xMu=acos(xTrace/2);
 cout << "  acos(xTrace/2)      " << acos(xTrace/2)        << "\n";
 cout << "  acos(xTrace/2)/2/PI " << acos(xTrace/2)/2/PI   << "\n";
 cout << "1+acos(xTrace/2)/2/PI " << 1+acos(xTrace/2)/2/PI << "\n";
 return 0;
}
