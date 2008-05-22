// Library       : AIM
// File          : AIM/BTF/BTFKicker.cc
// Copyright     : see Copyright file
// Author        : P.Cameron
// C++ version   : N.Malitsky 

#include <math.h>
#include "PAC/Beam/Bunch.hh"
#include "AIM/BTF/BTFKicker.hh"

AIM::BTFKicker::BTFKicker()
{
  init();
}

AIM::BTFKicker::BTFKicker(const BTFKicker& kicker)
{
  copy(kicker);
}

UAL::PropagatorNode* AIM::BTFKicker::clone()
{
  return new AIM::BTFKicker(*this);
}

void AIM::BTFKicker::setHKick(double hKick, double hNFreq, double hFracFreq, double hLag)
{
  m_hKick     = hKick;
  m_hNFreq    = hNFreq;
  m_hFracFreq = hFracFreq;
  m_hLag      = hLag;
}

void AIM::BTFKicker::setVKick(double vKick, double vNFreq, double vFracFreq, double vLag)
{
  m_vKick     = vKick;
  m_vNFreq    = vNFreq;
  m_vFracFreq = vFracFreq;
  m_vLag      = vLag;
}

void AIM::BTFKicker::setTurn(int turn)
{
  m_turn = turn;
}

void AIM::BTFKicker::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);

  double revFreq = bunch.getBeamAttributes().getRevfreq(); 

  // double t = bunch.getElapsedTime();
  // int turn = BTF::TurnCounter::getInstance().getTurn(); 

  double lagx     = 2.0*UAL::pi*(m_hLag   + m_hFracFreq*m_turn);
  double rlambdax = 2.0*UAL::pi*(m_hNFreq + m_hFracFreq)*revFreq/UAL::clight;

  double lagy     = 2.0*UAL::pi*(m_vLag   + m_vFracFreq*m_turn);
  double rlambday = 2.0*UAL::pi*(m_vNFreq + m_vFracFreq)*revFreq/UAL::clight;

  double vsinx, vsiny, ct;

  for(int ip = 0 ; ip < bunch.size() ; ip++) {

     if(!bunch[ip].isLost()) {

        PAC::Position& pos = bunch[ip].getPosition();

	ct = pos.getCT();

        vsinx = sin(lagx - rlambdax*ct);
        vsiny = sin(lagy - rlambday*ct);

        pos.setPX(pos.getPX() + m_hKick*vsinx);
        pos.setPY(pos.getPY() + m_vKick*vsiny); 

     }
  }

}

void AIM::BTFKicker::init()
{
  m_hKick     = 0.0;
  m_hNFreq    = 0.0;
  m_hFracFreq = 0.0;
  m_hLag      = 0.0;

  m_vKick     = 0.0;
  m_vNFreq    = 0.0;
  m_vFracFreq = 0.0;
  m_vLag      = 0.0;

  m_turn      = 0;
}

void AIM::BTFKicker::copy(const AIM::BTFKicker& kicker)
{
  m_hKick     = kicker.m_hKick;
  m_hNFreq    = kicker.m_hNFreq;
  m_hFracFreq = kicker.m_hFracFreq;
  m_hLag      = kicker.m_hLag;

  m_vKick     = kicker.m_vKick;
  m_vNFreq    = kicker.m_vNFreq;
  m_vFracFreq = kicker.m_vFracFreq;
  m_vLag      = kicker.m_vLag;

  m_turn      = kicker.m_turn;
}


