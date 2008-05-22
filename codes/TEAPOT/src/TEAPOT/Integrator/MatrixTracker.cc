
#include <math.h>
#include "UAL/APF/PropagatorFactory.hh"
#include "TEAPOT/Integrator/MatrixTracker.hh"

PacLattice TEAPOT::MatrixTracker::s_lattice;
Teapot     TEAPOT::MatrixTracker::s_teapot;

TEAPOT::MatrixTracker::MatrixTracker()
  : TEAPOT::BasicTracker()
{
  init();
}

TEAPOT::MatrixTracker::MatrixTracker(const TEAPOT::MatrixTracker& st)
  : TEAPOT::BasicTracker(st)
{
  copy(st);
}

TEAPOT::MatrixTracker::~MatrixTracker()
{
}

UAL::PropagatorNode* TEAPOT::MatrixTracker::clone()
{
  return new TEAPOT::MatrixTracker(*this);
}

void TEAPOT::MatrixTracker::setLatticeElements(const UAL::AcceleratorNode& sequence, int is0, int is1,
						const UAL::AttributeSet& attSet)
{
  TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);
  const PacLattice& lattice     = (PacLattice&) sequence;
  PAC::BeamAttributes ba = (PAC::BeamAttributes&) attSet;

  // Initialize lattice 
  if(s_lattice.name() != lattice.name()){
    s_lattice = lattice;
    s_teapot.use(lattice);
  }

  // Calculate length
  m_l = 0;
  for(int i = is0; i < is1; i++){
    m_l += lattice[i].getLength(); 
  }

  // Propagate the sector map
  PacTMap sectorMap(6);
  int oldMltOrder = sectorMap.mltOrder();
  sectorMap.mltOrder(1);
  s_teapot.trackMap(sectorMap, ba, is0, is1); 
  sectorMap.mltOrder(oldMltOrder);

  setMap(sectorMap);
}


void TEAPOT::MatrixTracker::setMap(const PacVTps& vtps)
{
  a10 = vtps(0, 0); a11 = vtps(0, 1); a12 = vtps(0, 2); a13 = vtps(0, 3); a14 = vtps(0, 4); a15 = vtps(0, 5); a16 = vtps(0, 6);
  a20 = vtps(1, 0); a21 = vtps(1, 1); a22 = vtps(1, 2); a23 = vtps(1, 3); a24 = vtps(1, 4); a25 = vtps(1, 5); a26 = vtps(1, 6);
  a30 = vtps(2, 0); a31 = vtps(2, 1); a32 = vtps(2, 2); a33 = vtps(2, 3); a34 = vtps(2, 4); a35 = vtps(2, 5); a36 = vtps(2, 6);  
  a40 = vtps(3, 0); a41 = vtps(3, 1); a42 = vtps(3, 2); a43 = vtps(3, 3); a44 = vtps(3, 4); a45 = vtps(3, 5); a46 = vtps(3, 6); 
  a50 = vtps(4, 0); a51 = vtps(4, 1); a52 = vtps(4, 2); a53 = vtps(4, 3); a54 = vtps(4, 4); a55 = vtps(4, 5); a56 = vtps(4, 6);   
  a60 = vtps(5, 0); a61 = vtps(5, 1); a62 = vtps(5, 2); a63 = vtps(5, 3); a64 = vtps(5, 4); a65 = vtps(5, 5); a66 = vtps(5, 6);  

}

void TEAPOT::MatrixTracker::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch& >(probe);

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double e0    = ba.getEnergy(), m0 = ba.getMass();
  double p0    = sqrt(e0*e0 - m0*m0);
  double v0byc = p0/e0;
  double oldT  = ba.getElapsedTime();

  double x0, px0, y0, py0, ct0, de0;
  double x, px, y, py, ct, de;
  double sx0, sy0, t0;
  // std::cout << "/n/n counter = " << s_counter++ << std::endl;

  int size = bunch.size();
  for(int i=0; i < size; i++){
    // if(bunch[i].getFlag() > 0 ) std::cout << "i = " << i << " lost " << std::endl;
    if(bunch[i].isLost()) continue;

    PAC::Position& pos = bunch[i].getPosition();
    
    x0  = pos.getX();
    px0 = pos.getPX();
    y0  = pos.getY();
    py0 = pos.getPY();
    ct0 = pos.getCT();
    de0 = pos.getDE();

    // calculate slops

    t0  = 1. - de0/v0byc ;
    sx0  = px0*t0; 
    sy0  = py0*t0;   
        
    x   = a10 + a11*x0 + a12*sx0 + a13*y0 + a14*sy0 + a15*ct0 + a16*de0;
    px  = a20 + a21*x0 + a22*px0 + a23*y0 + a24*py0 + a25*ct0 + a26*de0;
    y   = a30 + a31*x0 + a32*sx0 + a33*y0 + a34*sy0 + a35*ct0 + a36*de0;
    py  = a40 + a41*x0 + a42*px0 + a43*y0 + a44*py0 + a45*ct0 + a46*de0;
    ct  = a50 + a51*x0 + a52*px0 + a53*y0 + a54*py0 + a55*ct0 + a56*de0; 
    de  = a60 + a61*x0 + a62*px0 + a63*y0 + a64*py0 + a65*ct0 + a66*de0;

    pos.set(x, px, y, py, ct, de);
    
  }

  ba.setElapsedTime(oldT + m_l/v0byc/UAL::clight);  

}

void TEAPOT::MatrixTracker::init()
{
  // m_l = 0.0;

  a10 = a11 = a12 = a13 = a14 = a15 = a16 = 0.0;
  a20 = a21 = a22 = a23 = a24 = a25 = a26 = 0.0;
  a30 = a31 = a32 = a33 = a34 = a35 = a36 = 0.0;  
  a40 = a41 = a42 = a43 = a44 = a45 = a46 = 0.0; 
  a50 = a51 = a52 = a53 = a54 = a55 = a56 = 0.0;   
  a60 = a61 = a62 = a63 = a64 = a65 = a66 = 0.0;  
}

void TEAPOT::MatrixTracker::copy(const TEAPOT::MatrixTracker& mt)
{
  // m_l = mt.m_l;

  a10 = mt.a10; a11 = mt.a11; a12 = mt.a12; a13 = mt.a13; a14 = mt.a14; a15 = mt.a15; a16 = mt.a16;
  a20 = mt.a20; a21 = mt.a21; a22 = mt.a22; a23 = mt.a23; a24 = mt.a24; a25 = mt.a25; a26 = mt.a26;
  a30 = mt.a30; a31 = mt.a31; a32 = mt.a32; a33 = mt.a33; a34 = mt.a34; a35 = mt.a35; a36 = mt.a36;  
  a40 = mt.a40; a41 = mt.a41; a42 = mt.a42; a43 = mt.a43; a44 = mt.a44; a45 = mt.a45; a46 = mt.a46; 
  a50 = mt.a50; a51 = mt.a51; a52 = mt.a52; a53 = mt.a53; a54 = mt.a54; a55 = mt.a55; a56 = mt.a56;   
  a60 = mt.a60; a61 = mt.a61; a62 = mt.a62; a63 = mt.a63; a64 = mt.a64; a65 = mt.a65; a66 = mt.a66;  

}





