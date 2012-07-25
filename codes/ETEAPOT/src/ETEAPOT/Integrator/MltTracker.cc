// Library       : ETEAPOT
// File          : ETEAPOT/Integrator/MltTracker.cc
// Copyright     : see Copyright file
// Author        : L.Schachinger and R.Talman
// C++ version   : N.Malitsky 

#include <math.h>
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "ETEAPOT/Integrator/MltTracker.hh"
#include "ETEAPOT/Integrator/DipoleTracker.hh"

ETEAPOT::MltAlgorithm<double, PAC::Position> ETEAPOT::MltTracker::s_algorithm;
double ETEAPOT::MltTracker::m_m;
   int ETEAPOT::MltTracker::mltK=0;

ETEAPOT::MltTracker::MltTracker()
  : ETEAPOT::BasicTracker()
{
/*
  string line;
  ifstream m_m;
  m_m.open ("m_m");
  getline (m_m,line);
  ETEAPOT::MltTracker::m_m = atof( line.c_str() );
//std::cerr << "ETEAPOT::MltTracker::m_m " << ETEAPOT::MltTracker::m_m << "\n";
  m_m.close();
*/

  initialize();
}

ETEAPOT::MltTracker::MltTracker(const ETEAPOT::MltTracker& mt)
  : ETEAPOT::BasicTracker(mt)
{
  copy(mt);
}

ETEAPOT::MltTracker::~MltTracker()
{
}

UAL::PropagatorNode* ETEAPOT::MltTracker::clone()
{
  return new ETEAPOT::MltTracker(*this);
}

void ETEAPOT::MltTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					    int is0, 
					    int is1,
					    const UAL::AttributeSet& attSet)
{
   ETEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
   const PacLattice& lattice     = (PacLattice&) sequence;
   setLatticeElement(lattice[is0]);
}

void ETEAPOT::MltTracker::setLatticeElement(const PacLattElement& e)
{
  // length
  // m_l = e.getLength();

  // ir
  m_ir = e.getN();

  m_mdata.setLatticeElement(e);

}

void ETEAPOT::MltTracker::propagate(UAL::Probe& probe)
{
/*
  string line;
  ifstream m_m;
  m_m.open ("m_m");
  getline (m_m,line);
  ETEAPOT::MltTracker::m_m = atof( line.c_str() );
//std::cerr << "ETEAPOT::MltTracker::m_m " << ETEAPOT::MltTracker::m_m << "\n";
  m_m.close();
*/

  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);
  double v0byc = p0/e0;

  double oldT = ba.getElapsedTime();

  PAC::Position tmp;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

    s_algorithm.passEntry(ip, m_mdata, p, ETEAPOT::MltTracker::mltK, ETEAPOT::DipoleTracker::m_m );
//  s_algorithm.passEntry(m_mdata, p, ETEAPOT::MltTracker::mltK, ETEAPOT::MltTracker::m_m );
//  s_algorithm.passEntry(m_mdata, p, ETEAPOT::MltTracker::m_m );
//  s_algorithm.passEntry(m_mdata, p);

    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);

    // Simple Element

    if(!m_ir){
      s_algorithm.passDrift(m_l/2., p, tmp, v0byc);
      s_algorithm.applyMltKick(ip, m_mdata, 1., p, ETEAPOT::MltTracker::mltK, ETEAPOT::DipoleTracker::m_m );
//    s_algorithm.applyMltKick(m_mdata, 1., p, ETEAPOT::MltTracker::mltK, ETEAPOT::MltTracker::m_m );
//    s_algorithm.applyMltKick(m_mdata, 1., p, ETEAPOT::MltTracker::m_m );
//    s_algorithm.applyMltKick(m_mdata, 1., p);
      s_algorithm.makeVelocity(p, tmp, v0byc);
      s_algorithm.passDrift(m_l/2., p, tmp, v0byc);
      s_algorithm.passExit(ip, m_mdata, p, ETEAPOT::MltTracker::mltK, ETEAPOT::DipoleTracker::m_m );
//    s_algorithm.passExit(m_mdata, p, ETEAPOT::MltTracker::mltK, ETEAPOT::MltTracker::m_m );
//    s_algorithm.passExit(m_mdata, p);
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
          s_algorithm.passDrift(m_l*s_steps[is]*rIr, p, tmp, v0byc);
          s_algorithm.applyMltKick(ip, m_mdata, rkicks, p, ETEAPOT::MltTracker::mltK, ETEAPOT::DipoleTracker::m_m );
//        s_algorithm.applyMltKick(m_mdata, rkicks, p, ETEAPOT::MltTracker::mltK, ETEAPOT::MltTracker::m_m );
//        s_algorithm.applyMltKick(m_mdata, rkicks, p, ETEAPOT::MltTracker::m_m );
//        s_algorithm.applyMltKick(m_mdata, rkicks, p);
          s_algorithm.makeVelocity(p, tmp, v0byc);	
        }
        counter++;
        s_algorithm.passDrift(m_l*s_steps[4]*rIr, p, tmp, v0byc); 
      }
    }

    s_algorithm.passExit(ip, m_mdata, p, ETEAPOT::MltTracker::mltK, ETEAPOT::DipoleTracker::m_m );
//  s_algorithm.passExit(m_mdata, p, ETEAPOT::MltTracker::mltK, ETEAPOT::MltTracker::m_m );
//  s_algorithm.passExit(m_mdata, p);
    // testAperture(p);
  }
mltK++;

  checkAperture(bunch);

  ba.setElapsedTime(oldT + m_l/v0byc/UAL::clight);  

}

void ETEAPOT::MltTracker::initialize()
{
  // m_l = 0.0;
  m_ir = 0.0;
}

void ETEAPOT::MltTracker::copy(const ETEAPOT::MltTracker& mt)
{
  // m_l   = mt.m_l;
  m_ir  = mt.m_ir;

  m_mdata = mt.m_mdata;
}










