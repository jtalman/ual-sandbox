// Library       : AIM
// File          : AIM/BTF/BTFBpmCollector.cc
// Copyright     : see Copyright file
// Author        : N.Malitsky 

#include "AIM/BTF/BTFBpmCollector.hh"

AIM::BTFBpmCollector* AIM::BTFBpmCollector::s_theInstance = 0;

AIM::BTFBpmCollector::BTFBpmCollector()
{
}


void AIM::BTFBpmCollector::clear()
{
  m_signals.clear();
  m_hSpecs.clear();
  m_vSpecs.clear();
}


AIM::BTFBpmCollector& AIM::BTFBpmCollector::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new AIM::BTFBpmCollector();
  }
  return *s_theInstance;
}

void AIM::BTFBpmCollector::addSignal(AIM::BTFSignal& signal)
{
  m_signals.push_back(signal);
}

void AIM::BTFBpmCollector::addHSpectrum(AIM::BTFSpectrum& spec)
{
  m_hSpecs.push_back(spec);
}

void AIM::BTFBpmCollector::addVSpectrum(AIM::BTFSpectrum& spec)
{
  m_vSpecs.push_back(spec);
}

const std::list<AIM::BTFSignal>& AIM::BTFBpmCollector::getSignals()
{
  return m_signals;
}

const std::list<AIM::BTFSpectrum>& AIM::BTFBpmCollector::getHSpectrum()
{
  return m_hSpecs;
}

const std::list<AIM::BTFSpectrum>& AIM::BTFBpmCollector::getVSpectrum()
{
  return m_vSpecs;
}
