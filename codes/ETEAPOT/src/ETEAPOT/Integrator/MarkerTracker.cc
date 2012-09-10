// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/MarkerTracker.cc
// Copyright     : see Copyright file


#include <math.h>
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacLattice.h"
#include "ETEAPOT/Integrator/MarkerTracker.hh"
#include "ETEAPOT/Integrator/DipoleTracker.hh"

//#include "UAL/UI/OpticsCalculator.hh"
//#include "Main/Teapot.h"
#include <cstdlib>

//ETEAPOT::MarkerAlgorithm<double, PAC::Position> ETEAPOT::MarkerTracker::s_algorithm;
int ETEAPOT::MarkerTracker::mark=0;
std::string ETEAPOT::MarkerTracker::Mark_m_elementName[1000];
double ETEAPOT::MarkerTracker::Mark_m_sX[1000];

ETEAPOT::MarkerTracker::MarkerTracker()
  : ETEAPOT::BasicTracker()
{
}

ETEAPOT::MarkerTracker::MarkerTracker(const ETEAPOT::MarkerTracker& dt)
  : ETEAPOT::BasicTracker(dt)
{
}

ETEAPOT::MarkerTracker::~MarkerTracker()
{
}

UAL::PropagatorNode* ETEAPOT::MarkerTracker::clone()
{
  return new ETEAPOT::MarkerTracker(*this);
}

void ETEAPOT::MarkerTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					      int is0, 
					      int is1,
					      const UAL::AttributeSet& attSet)
{
   ETEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
}

void ETEAPOT::MarkerTracker::propagate(UAL::Probe& probe)
{
//std::cerr << "File " << __FILE__ << " line " << __LINE__ << " method void ETEAPOT::MarkerTracker::propagate(UAL::Probe& probe)\n";

  char * S[21] = {"ZERO  ","ONE   ","TWO   ","THREE ","FOUR  ","FIVE  ","SIX   ","SEVEN ","EIGHT ","NINE  ","TEN   ","ELEVEN","TWELVE","THIRTN","FORTN ","FIFTN ","SIKTN ","SEVNTN","EGHTN ","NNETN ","TWENTY"};

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
//  s_algorithm.passMarkerPlusPostProcess( ip, m_l, p, tmp, v0byc, ETEAPOT::DipoleTracker::m_m, drft );

#ifndef lngTrmTrk
char buffer [3];
sprintf(buffer,"%d",ip);
std::string bip(buffer);
char buffr2 [10];
sprintf(buffr2,"%+5.2f",ETEAPOT::DipoleTracker::m_m);
std::string bp2(buffr2);
std::string sip = "out/TWISS/StndrdPrtcl";
            sip+=bip;
std::cout << "sip.length() " << sip.length() << "\n";
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
  filestr << ETEAPOT::MarkerTracker::Mark_m_elementName[mark] << " " << ETEAPOT::MarkerTracker::Mark_m_sX[mark] << " " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << " " << "mark" << setw(5) << mark << " " << S[ip] << "\n";
//filestr << "MARKER  " << mark << " " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << " " << "mark" << setw(5) << mark << " " << S[ip] << "\n";
filestr.close();
#endif

  }
mark++;

//checkAperture(bunch);

//ba.setElapsedTime(oldT + m_l/v0byc/UAL::clight);
}
