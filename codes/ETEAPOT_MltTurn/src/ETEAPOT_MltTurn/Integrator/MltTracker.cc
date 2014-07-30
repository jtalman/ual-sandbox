#include <math.h>
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
//#include "ETEAPOT/Integrator/MltTracker.hh"

#include "ETEAPOT_MltTurn/Integrator/MltTracker.hh"
#include "ETEAPOT_MltTurn/Integrator/DipoleTracker.hh"

ETEAPOT_MltTurn::MltAlgorithm<double, PAC::Position> ETEAPOT_MltTurn::MltTracker::s_algorithm;
double ETEAPOT_MltTurn::MltTracker::m_m;
   int ETEAPOT_MltTurn::MltTracker::mltK=0;
double ETEAPOT_MltTurn::MltTracker::spin[41][3];

ETEAPOT_MltTurn::MltTracker::MltTracker()
  : ETEAPOT::BasicTracker()
{
//initialize();
  m_ir = 0.0;
}

ETEAPOT_MltTurn::MltTracker::MltTracker(const ETEAPOT_MltTurn::MltTracker& mt)
  : ETEAPOT::BasicTracker(mt)
{
  copy(mt);
}

ETEAPOT_MltTurn::MltTracker::~MltTracker()
{
}

UAL::PropagatorNode* ETEAPOT_MltTurn::MltTracker::clone()
{
  return new ETEAPOT_MltTurn::MltTracker(*this);
}

void ETEAPOT_MltTurn::MltTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					    int is0, 
					    int is1,
					    const UAL::AttributeSet& attSet)
{
   ETEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void ETEAPOT_MltTurn::MltTracker::setLatticeElement(const PacLattElement& e)
{
  // length
  // m_l = e.getLength();

  // ir
  m_ir = e.getN();

  m_mdata.setLatticeElement(e);

}

void ETEAPOT_MltTurn::MltTracker::propagate(UAL::Probe& probe)
{
/*
  string line;
  ifstream m_m;
  m_m.open ("m_m");
  getline (m_m,line);
  ETEAPOT_MltTurn::MltTracker::m_m = atof( line.c_str() );
//std::cerr << "ETEAPOT_MltTurn::MltTracker::m_m " << ETEAPOT_MltTurn::MltTracker::m_m << "\n";
  m_m.close();
*/

  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  const PAC::BeamAttributes cba = ba;

  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);
  double v0byc = p0/e0;

  double oldT = ba.getElapsedTime();

  PAC::Position tmp;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

    ETEAPOT_MltTurn::MltTracker::s_algorithm.passEntry(ip, m_mdata, p, ETEAPOT_MltTurn::MltTracker::mltK, ETEAPOT_MltTurn::MltTracker::m_m, cba );
//  ETEAPOT_MltTurn::MltTracker::s_algorithm.passEntry(m_mdata, p, ETEAPOT_MltTurn::MltTracker::mltK, ETEAPOT_MltTurn::MltTracker::m_m );
//  ETEAPOT_MltTurn::MltTracker::s_algorithm.passEntry(m_mdata, p, ETEAPOT_MltTurn::MltTracker::m_m );
//  ETEAPOT_MltTurn::MltTracker::s_algorithm.passEntry(m_mdata, p);

    ETEAPOT_MltTurn::MltTracker::s_algorithm.makeVelocity(p, tmp, v0byc);
    ETEAPOT_MltTurn::MltTracker::s_algorithm.makeRV(p, tmp, e0, p0, m0);

    // Simple Element

    if(!m_ir){
      ETEAPOT_MltTurn::MltTracker::s_algorithm.passDrift(m_l/2., p, tmp, v0byc);
      ETEAPOT_MltTurn::MltTracker::s_algorithm.applyMltKick(ip, m_mdata, 1., p, ETEAPOT_MltTurn::MltTracker::mltK, ETEAPOT_MltTurn::MltTracker::m_m, cba );
//    ETEAPOT_MltTurn::MltTracker::s_algorithm.applyMltKick(m_mdata, 1., p, ETEAPOT_MltTurn::MltTracker::mltK, ETEAPOT_MltTurn::MltTracker::m_m );
//    ETEAPOT_MltTurn::MltTracker::s_algorithm.applyMltKick(m_mdata, 1., p, ETEAPOT_MltTurn::MltTracker::m_m );
//    ETEAPOT_MltTurn::MltTracker::s_algorithm.applyMltKick(m_mdata, 1., p);
      ETEAPOT_MltTurn::MltTracker::s_algorithm.makeVelocity(p, tmp, v0byc);
      ETEAPOT_MltTurn::MltTracker::s_algorithm.passDrift(m_l/2., p, tmp, v0byc);
      ETEAPOT_MltTurn::MltTracker::s_algorithm.passExit(ip, m_mdata, p, ETEAPOT_MltTurn::MltTracker::mltK, ETEAPOT_MltTurn::MltTracker::m_m, cba );
//    ETEAPOT_MltTurn::MltTracker::s_algorithm.passExit(m_mdata, p, ETEAPOT_MltTurn::MltTracker::mltK, ETEAPOT_MltTurn::MltTracker::m_m );
//    ETEAPOT_MltTurn::MltTracker::s_algorithm.passExit(m_mdata, p);
      continue;
    } 
    else{
      std::cerr << "Complex Elements not allowed!!!\n";
      exit(1);

      // Complex Element

      double rIr = 1./m_ir;
      double rkicks = 0.25*rIr;

      int counter = 0;
      for(int i = 0; i < m_ir; i++){
        for(int is = 0; is < 4; is++){
          counter++;
          ETEAPOT_MltTurn::MltTracker::s_algorithm.passDrift(m_l*s_steps[is]*rIr, p, tmp, v0byc);
          ETEAPOT_MltTurn::MltTracker::s_algorithm.applyMltKick(ip, m_mdata, rkicks, p, ETEAPOT_MltTurn::MltTracker::mltK, ETEAPOT_MltTurn::MltTracker::m_m, cba );
//        ETEAPOT_MltTurn::MltTracker::s_algorithm.applyMltKick(m_mdata, rkicks, p, ETEAPOT_MltTurn::MltTracker::mltK, ETEAPOT_MltTurn::MltTracker::m_m );
//        ETEAPOT_MltTurn::MltTracker::s_algorithm.applyMltKick(m_mdata, rkicks, p, ETEAPOT_MltTurn::MltTracker::m_m );
//        ETEAPOT_MltTurn::MltTracker::s_algorithm.applyMltKick(m_mdata, rkicks, p);
          ETEAPOT_MltTurn::MltTracker::s_algorithm.makeVelocity(p, tmp, v0byc);	
        }
        counter++;
        ETEAPOT_MltTurn::MltTracker::s_algorithm.passDrift(m_l*s_steps[4]*rIr, p, tmp, v0byc); 
      }
    }

    ETEAPOT_MltTurn::MltTracker::s_algorithm.passExit(ip, m_mdata, p, ETEAPOT_MltTurn::MltTracker::mltK, ETEAPOT_MltTurn::MltTracker::m_m, cba );
//  ETEAPOT_MltTurn::MltTracker::s_algorithm.passExit(m_mdata, p, ETEAPOT_MltTurn::MltTracker::mltK, ETEAPOT_MltTurn::MltTracker::m_m );
//  ETEAPOT_MltTurn::MltTracker::s_algorithm.passExit(m_mdata, p);
    // testAperture(p);
  }
  for(int ip = 0; ip < bunch.size(); ip++) {
/*
   ETEAPOT_MltTurn::MltTracker::s_algorithm.spin[ip][0]=s_algorithm.spin[ip][0];
   ETEAPOT_MltTurn::MltTracker::s_algorithm.spin[ip][1]=s_algorithm.spin[ip][1];
   ETEAPOT_MltTurn::MltTracker::s_algorithm.spin[ip][2]=s_algorithm.spin[ip][2];
*/
  }
//#include"setDipoleTrackerSpin"
for(int ip=0;ip<=40;ip++){
 for(int iq=0;iq<=2;iq++){
//ETEAPOT_MltTurn::DipoleTracker::spin[ip][iq]=spin[ip][iq];
  ETEAPOT_MltTurn::DipoleTracker::s_algorithm.spin[ip][iq]=s_algorithm.spin[ip][iq];
 }
}
//#include"setMarkerTrackerSpin"
mltK++;

  checkAperture(bunch);

  ba.setElapsedTime(oldT + m_l/v0byc/UAL::clight);  

}

/*
void ETEAPOT_MltTurn::MltTracker::initialize()
{
  // m_l = 0.0;
  m_ir = 0.0;
}
*/

void ETEAPOT_MltTurn::MltTracker::copy(const ETEAPOT_MltTurn::MltTracker& mt)
{
  // m_l   = mt.m_l;
  m_ir  = mt.m_ir;

  m_mdata = mt.m_mdata;
}
