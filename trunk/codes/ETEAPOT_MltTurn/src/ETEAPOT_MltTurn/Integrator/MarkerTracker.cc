#include <math.h>
#include <string.h>
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
//#include "ETEAPOT/Integrator/DipoleTracker.hh"

#include "ETEAPOT_MltTurn/Integrator/MarkerTracker.hh"
#include "ETEAPOT_MltTurn/Integrator/DipoleTracker.hh"

//#include "UAL/UI/OpticsCalculator.hh"
//#include "Main/Teapot.h"
#include <cstdlib>

//ETEAPOT_MltTurn::MarkerAlgorithm<double, PAC::Position> ETEAPOT_MltTurn::MarkerTracker::s_algorithm;
int ETEAPOT_MltTurn::MarkerTracker::mark=0;
std::string ETEAPOT_MltTurn::MarkerTracker::Mark_m_elementName[1000];
double ETEAPOT_MltTurn::MarkerTracker::Mark_m_sX[1000];
//void ETEAPOT_MltTurn::MarkerTracker::initialize();
double ETEAPOT_MltTurn::MarkerTracker::spin[21][3];

ETEAPOT_MltTurn::MarkerTracker::MarkerTracker()
  : ETEAPOT::BasicTracker()
{
}

ETEAPOT_MltTurn::MarkerTracker::MarkerTracker(const ETEAPOT_MltTurn::MarkerTracker& dt)
  : ETEAPOT::BasicTracker(dt)
{
}

ETEAPOT_MltTurn::MarkerTracker::~MarkerTracker()
{
}

UAL::PropagatorNode* ETEAPOT_MltTurn::MarkerTracker::clone()
{
  return new ETEAPOT_MltTurn::MarkerTracker(*this);
}

void ETEAPOT_MltTurn::MarkerTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					      int is0, 
					      int is1,
					      const UAL::AttributeSet& attSet)
{
   ETEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
}

void ETEAPOT_MltTurn::MarkerTracker::propagate(UAL::Probe& probe)
{
//std::cerr << "ENTER: FILE: " << __FILE__ << " LINE: " << __LINE__ << " method void ETEAPOT_MltTurn::MarkerTracker::propagate(UAL::Probe& probe)\n";
  char * S[21] = {"ZERO  ","ONE   ","TWO   ","THREE ","FOUR  ","FIVE  ","SIX   ","SEVEN ","EIGHT ","NINE  ","TEN   ","ELEVEN","TWELVE","THIRTN","FORTN ","FIFTN ","SIKTN ","SEVNTN","EGHTN ","NNETN ","TWENTY"};

//std::string MM = ETEAPOT_MltTurn::MarkerTracker::Mark_m_elementName[mark];
  std::string MM = "mdummy";
  const char * MMp = MM.c_str();
  if( strcmp( "mbegin      ", MMp ) == 0 ){
   ifstream spinIFS;
   spinIFS.open ("initialSpin", ifstream::in);

   ofstream spinOFS;
   spinOFS.open ("out/VERIF/initialSpin");
   spinOFS << setiosflags( ios::showpos    );  
   spinOFS << setiosflags( ios::uppercase  );  
   spinOFS << setiosflags( ios::scientific );
   //spinOFS << setw( 11 );
   spinOFS << setfill( ' ' );
   spinOFS << setiosflags( ios::left );
   spinOFS << setprecision(13) ;

   PAC::Spin echo;
   std::string spinX,spinY,spinZ;
// while( !spinIFS.eof() ){
   int ip=-1;
   std::string Name;
   while( 1              ){
    ip++;
    spinIFS >> Name >> spinX >> spinY >> spinZ;
    echo.setSX( atof(spinX.c_str()) );echo.setSY( atof(spinY.c_str()) );echo.setSZ( atof(spinZ.c_str()) );

    if( !spinIFS.eof() ){
     spinOFS << S[ip] << " " << echo.getSX() << " " << echo.getSY() << " " << echo.getSZ() << "\n";
//   spinOFS << Name << " " << echo.getSX() << " " << echo.getSY() << " " << echo.getSZ() << "\n";
    }
    else{
     spinOFS << S[ip] << " " << echo.getSX() << " " << echo.getSY() << " " << echo.getSZ();
     break;
    }
   }

   spinIFS.close();
   spinOFS.close();

   std::cerr << "mbegin, ETEAPOT_MltTurn::MarkerTracker::Mark_m_elementName[mark] == 0 \n";
  }
  else{
// std::cerr << "mbegin, ETEAPOT_MltTurn::MarkerTracker::Mark_m_elementName[mark] != 0 \n";
  }

//char * S[21] = {"ZERO  ","ONE   ","TWO   ","THREE ","FOUR  ","FIVE  ","SIX   ","SEVEN ","EIGHT ","NINE  ","TEN   ","ELEVEN","TWELVE","THIRTN","FORTN ","FIFTN ","SIKTN ","SEVNTN","EGHTN ","NNETN ","TWENTY"};

  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();

  double oldT = ba.getElapsedTime();

  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);

  double v0byc = p0/e0;

  PAC::Position tmp;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;
//  s_algorithm.makeVelocity(p, tmp, v0byc);
//  s_algorithm.makeRV(p, tmp, e0, p0, m0);
//  s_algorithm.passMarkerPlusPostProcess( ip, m_l, p, tmp, v0byc, ETEAPOT_MltTurn::DipoleTracker::m_m, drft );

#ifndef lngTrmTrk
char buffer [3];
sprintf(buffer,"%d",ip);
std::string bip(buffer);
char buffr2 [10];
sprintf(buffr2,"%+5.2f",ETEAPOT_MltTurn::DipoleTracker::m_m);
std::string bp2(buffr2);
std::string sip = "out/TWISS/StndrdPrtcl";
            sip+=bip;
//std::cout << "sip.length() " << sip.length() << "\n";
/*
if(sip.length()==22){sip+="_";}
            sip+="_m=";
            sip+=bp2;
*/
fstream filestr;
filestr.open (sip.c_str(), fstream::out | fstream::app);
filestr << setiosflags( ios::showpos    );  
filestr << setiosflags( ios::uppercase  );  
filestr << setiosflags( ios::scientific );
filestr << setfill( ' ' );
filestr << setiosflags( ios::left );
filestr << setprecision(13) ;
//filestr << ETEAPOT_MltTurn::MarkerTracker::Mark_m_elementName[mark] << " " << ETEAPOT_MltTurn::MarkerTracker::Mark_m_sX[mark] << " " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << " " << "mark" << setw(5) << mark << " " << S[ip] << "\n";
//filestr << "MARKER  " << mark << " " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << " " << "mark" << setw(5) << mark << " " << S[ip] << "\n";
filestr.close();
#endif

  }
mark++;

//checkAperture(bunch);

//ba.setElapsedTime(oldT + m_l/v0byc/UAL::clight);
}

/*
static void initialize(){
 std::cerr << "enter static void initialize(){ \n";
 char * S[21] = {"ZERO  ","ONE   ","TWO   ","THREE ","FOUR  ","FIVE  ","SIX   ","SEVEN ","EIGHT ","NINE  ","TEN   ","ELEVEN","TWELVE","THIRTN","FORTN ","FIFTN ","SIKTN ","SEVNTN","EGHTN ","NNETN ","TWENTY"};

 ifstream spinIFS;
 spinIFS.open ("initialSpin", ifstream::in);

 ofstream spinOFS;
 spinOFS.open ("out/VERIF/initialSpin");
 spinOFS << setiosflags( ios::showpos    );  
 spinOFS << setiosflags( ios::uppercase  );  
 spinOFS << setiosflags( ios::scientific );
 spinOFS << setfill( ' ' );
 spinOFS << setiosflags( ios::left );
 spinOFS << setprecision(13) ;

 PAC::Spin echo;
 std::string spinX,spinY,spinZ;
 int ip=-1;
 std::string Name;
 while(1){
  ip++;
  spinIFS >> Name >> spinX >> spinY >> spinZ;
  echo.setSX( atof(spinX.c_str()) );echo.setSY( atof(spinY.c_str()) );echo.setSZ( atof(spinZ.c_str()) );

  if( !spinIFS.eof() ){
   spinOFS << S[ip] << " " << echo.getSX() << " " << echo.getSY() << " " << echo.getSZ() << "\n";
  }
  else{
   spinOFS << S[ip] << " " << echo.getSX() << " " << echo.getSY() << " " << echo.getSZ();
   break;
  }
 }

 spinIFS.close();
 spinOFS.close();
}
*/
