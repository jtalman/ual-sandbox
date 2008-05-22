// Library       : SIMBAD
// File          : SIMBAD/SC/TSCPropagatorFFT.cc
// Copyright     : see Copyright file
// Author        : N.D'Imperio, A.Luccio et al.

#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "SIMBAD/Tracker/MatrixTracker.hh"
#include "SIMBAD/SC/TSCPropagatorFFT.hh"
#include "SIMBAD/SC/TSCCalculatorFFT.hh"

using namespace std;


SIMBAD::TSCPropagatorFFT::TSCPropagatorFFT()
{
  m_lkick = 0.0;
}

SIMBAD::TSCPropagatorFFT::TSCPropagatorFFT(const SIMBAD::TSCPropagatorFFT& p)
{
  m_lkick   = p.m_lkick;
  m_tracker = p.m_tracker;
}

SIMBAD::TSCPropagatorFFT::~TSCPropagatorFFT()
{
}

UAL::PropagatorNode* SIMBAD::TSCPropagatorFFT::clone()
{
  return new SIMBAD::TSCPropagatorFFT(*this);
}

void SIMBAD::TSCPropagatorFFT::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						  int is0, int is1,
						  const UAL::AttributeSet& attSet)
{

  const PacLattice& lattice = (PacLattice&) sequence;

  // Defines nodes
  SIMBAD::BasicPropagator::setLatticeElements(sequence, is0, is1, attSet);

  // Set a length
  m_lkick = lattice[is0].getLength();

  // Set a conventional tracker


  UAL::PropagatorNodePtr nodePtr = 
    TEAPOT::TrackerFactory::createTracker(lattice[is0].getType());

  // UAL::PropagatorNodePtr nodePtr = new SIMBAD::MatrixTracker();

  m_tracker = nodePtr;

  // Set tracker data
  m_tracker->setLatticeElements(sequence, is0, is1, attSet);
}

void SIMBAD::TSCPropagatorFFT::setLength(double l)
{
  m_lkick = l;
}

double SIMBAD::TSCPropagatorFFT::getLength() const
{
  return m_lkick;
}

void SIMBAD::TSCPropagatorFFT::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
  if(m_tracker.isValid()) m_tracker->propagate(bunch);

  SIMBAD::TSCCalculatorFFT::getInstance().calculateForce(bunch);
  SIMBAD::TSCCalculatorFFT::getInstance().propagate(bunch, m_lkick);
}

SIMBAD::TSCPropagatorFFTRegister::TSCPropagatorFFTRegister()
{
  UAL::PropagatorNodePtr nodePtr(new SIMBAD::TSCPropagatorFFT());
  UAL::PropagatorFactory::getInstance().add("SIMBAD::TSCPropagatorFFT", nodePtr);
}

static SIMBAD::TSCPropagatorFFTRegister theSimbadTSCPropagatorFFTRegister; 


