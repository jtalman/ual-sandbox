// Library       : SXF_TRACKER
// File          : examples/SXF_TRACKER/driftOrDefault.cc
// Copyright     : see Copyright file
// Author        : 
// C++ version   : J.Talman, N.Malitsky

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "driftOrDefault.hh"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <fstream>

SXF_TRACKER::driftOrDefault::driftOrDefault()
{
  p_entryMlt = 0;
  p_exitMlt = 0;
  p_length = 0;
  p_bend = 0;
  p_mlt = 0;
  p_offset = 0;
  p_rotation = 0;
  // p_aperture = 0;
  p_complexity = 0;
  // p_solenoid = 0;
  // p_rf = 0;
}

SXF_TRACKER::driftOrDefault::driftOrDefault(const SXF_TRACKER::driftOrDefault& st)
{
  copy(st);
}

SXF_TRACKER::driftOrDefault::~driftOrDefault()
{
}

UAL::PropagatorNode* SXF_TRACKER::driftOrDefault::clone()
{
  return new SXF_TRACKER::driftOrDefault(*this);
}


void SXF_TRACKER::driftOrDefault::setLatticeElements(const UAL::AcceleratorNode& sequence,
					   int is0, int is1,
					   const UAL::AttributeSet& attSet)
{
//  UAL::PropagatorNode::setLatticeElements(sequence, is0, is1, attSet);

  if(is0 < sequence.getNodeCount())
    m_frontNode = *((PacLattElement*) sequence.getNodeAt(is0));
  if(is1 < sequence.getNodeCount()) 
    m_backNode  = *((PacLattElement*) sequence.getNodeAt(is1));
 
    const PacLattice& lattice = (PacLattice&) sequence;

    setElementData(lattice[is0]);
    setConventionalTracker(sequence, is0, is1, attSet);

    m_name = lattice[is0].getName();

    // std::cout << "driftOrDefault  "<<is0 << " " << lattice[is0].getName() << " " << lattice[is0].getType()  << std::endl;

   // if(p_complexity) std::cout << " n = " << p_complexity->n()  << std::endl;
   // if(p_length)  std::cout << " l = " << p_length->l() << std::endl;
   // if(p_bend)   std::cout <<  " angle = " << p_bend->angle() << std::endl;
   // if(p_mlt)    std::cout << " kl1 = "  << p_mlt->kl(1) << std::endl;
   // std::cout << std::endl;

}

void SXF_TRACKER::driftOrDefault::propagate(UAL::Probe& b)
{

  // SXF_TRACKER::driftOrDefaultWriter* stw = SXF_TRACKER::driftOrDefaultWriter::getInstance();

  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);

  // stw->write(bunch.getBeamAttributes().getElapsedTime());

  /*
  PAC::Position& pos = bunch[0].getPosition();

  double sx = bunch[0].getdriftOrDefault()->getSX();
  double sy = bunch[0].getdriftOrDefault()->getSY();
  double sz = bunch[0].getdriftOrDefault()->getSZ();
  double x  = pos.getX();
  double px = pos.getPX();
  double y  = pos.getY();
  double py = pos.getPY();
  double ct = pos.getCT();
  double de = pos.getDE();
  */

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();

  double energy = ba.getEnergy();
  double mass   = ba.getMass();
  double gam    = energy/mass;

  double p = sqrt(energy*energy - mass*mass);
  double v = p/gam/mass*UAL::clight;

  double t0 = ba.getElapsedTime();

  double length = 0;
  if(p_length)     length = p_length->l();

  if(!p_complexity){

      length /= 2;

    if(p_mlt) *p_mlt /= 2.;             // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= 2.;             // kl, kt

    t0 += length/v;
    ba.setElapsedTime(t0);

    propagateSXF_Tracker(b);

    if(p_mlt) *p_mlt /= 2.;             // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= 2.;             // kl, kt

    t0 += length/v;
    ba.setElapsedTime(t0);

    return;
  }

  int ns = (int)4*p_complexity->n();

  length /= 2*ns;

  for(int i=0; i < ns; i++) {

    if(p_mlt) *p_mlt /= (2*ns);          // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= (2*ns);          // kl, kt

    t0 += length/v;
    ba.setElapsedTime(t0);

    propagateSXF_Tracker(b);

    if(p_mlt) *p_mlt /= (2*ns);          // kl, kt
    m_tracker->propagate(bunch);
    if(p_mlt) *p_mlt *= (2*ns);          // kl, kt

    t0 += length/v;
    ba.setElapsedTime(t0);

  }


}

double SXF_TRACKER::driftOrDefault::get_psp0(PAC::Position& p, double v0byc)
{
    double psp0  = 1.0;  

    psp0 -= p.getPX()*p.getPX();
    psp0 -= p.getPY()*p.getPY();

    psp0 += p.getDE()*p.getDE();
    psp0 += (2./v0byc)*p.getDE();

    psp0 = sqrt(psp0);

    return psp0;
}

void SXF_TRACKER::driftOrDefault::setElementData(const PacLattElement& e)
{
 
  // Entry multipole
  PacElemAttributes* front  = e.getFront();
  if(front){
     PacElemAttributes::iterator it = front->find(PAC_MULTIPOLE);
     if(it != front->end()) p_entryMlt = (PacElemMultipole*) &(*it);
  }

  // Exit multipole
  PacElemAttributes* end  = e.getEnd();
  if(end){
     PacElemAttributes::iterator it = end->find(PAC_MULTIPOLE);
     if(it != end->end()) p_exitMlt = (PacElemMultipole*) &(*it);
  }

  // Body attributes
  PacElemAttributes* attributes = e.getBody();

  if(attributes){
    for(PacElemAttributes::iterator it = attributes->begin(); it != attributes->end(); it++){
      switch((*it).key()){
       case PAC_LENGTH:                          // 1: l
            p_length = (PacElemLength*) &(*it);
            break;
       case PAC_BEND:                            // 2: angle, fint
            p_bend = (PacElemBend*) &(*it);
            break;
       case PAC_MULTIPOLE:                       // 3: kl, ktl
            p_mlt = (PacElemMultipole*) &(*it);
            break;
       case PAC_OFFSET:                          // 4: dx, dy, ds
            p_offset = (PacElemOffset*) &(*it);
            break;
       case PAC_ROTATION:                        // 5: dphi, dtheta, tilt
            p_rotation = (PacElemRotation*) &(*it);
            break;
       case PAC_APERTURE:                        // 6: shape, xsize, ysize
	    // p_aperture = (PacElemAperture*) &(*it);
	    break;
       case PAC_COMPLEXITY:                     // 7: n
            p_complexity = (PacElemComplexity* ) &(*it);
            break;
       case PAC_SOLENOID:                       // 8: ks
            // p_solenoid = (PacElemSolenoid* ) &(*it);
            break;
       case PAC_RFCAVITY:                       // 9: volt, lag, harmon
           // p_rf = (PacElemRfCavity* ) &(*it);
           break;
      default:
	break;
      }
    }
  }

}

void SXF_TRACKER::driftOrDefault::setConventionalTracker(const UAL::AcceleratorNode& sequence,
                                                int is0, int is1,
                                                const UAL::AttributeSet& attSet)
{
    const PacLattice& lattice = (PacLattice&) sequence;

    double ns = 2;
    if(p_complexity) ns = 8*p_complexity->n();

    UAL::PropagatorNodePtr nodePtr =
      TEAPOT::TrackerFactory::createTracker(lattice[is0].getType());

    m_tracker = nodePtr;
    
    if(p_complexity) p_complexity->n() = 0;   // ir
    if(p_length)    *p_length /= ns;          // l
    if(p_bend)      *p_bend /= ns;            // angle, fint
     
    m_tracker->setLatticeElements(sequence, is0, is1, attSet);
     
    if(p_bend)      *p_bend *= ns;
    if(p_length)    *p_length *= ns;
    if(p_complexity) p_complexity->n() = ns/8;

}

void SXF_TRACKER::driftOrDefault::propagateSXF_Tracker(UAL::Probe& b)
{
    PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);

    PAC::BeamAttributes& ba = bunch.getBeamAttributes();

    for(int i=0; i < bunch.size(); i++){
        propagateSXF_Tracker(ba, bunch[i]);
    }
}

void SXF_TRACKER::driftOrDefault::propagateSXF_Tracker(PAC::BeamAttributes& ba, PAC::Particle& prt)
{
}

void SXF_TRACKER::driftOrDefault::copy(const SXF_TRACKER::driftOrDefault& st)
{
    m_name       = st.m_name;

    p_entryMlt   = st.p_entryMlt;
    p_exitMlt    = st.p_exitMlt;

    p_length     = st.p_length;
    p_bend       = st.p_bend;
    p_mlt        = st.p_mlt;
    p_offset     = st.p_offset;
    p_rotation   = st.p_rotation;
    // p_aperture = st.p_aperture;
    p_complexity = st.p_complexity;
    // p_solenoid = st.p_solenoid;
    // p_rf = st.p_rf;
}
  
SXF_TRACKER::driftOrDefaultRegister::driftOrDefaultRegister()
{
  UAL::PropagatorNodePtr driftOrDefaultPtr((UAL::PropagatorNode*)new driftOrDefault());
  UAL::PropagatorFactory::getInstance().add("SXF_TRACKER::driftOrDefault", driftOrDefaultPtr);
}

static SXF_TRACKER::driftOrDefaultRegister thedriftOrDefaultRegister;
