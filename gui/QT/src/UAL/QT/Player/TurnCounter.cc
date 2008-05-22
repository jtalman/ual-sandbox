#include "UAL/QT/Player/TurnCounter.hh"

UAL::QT::TurnCounter* UAL::QT::TurnCounter::s_theInstance = 0;

UAL::QT::TurnCounter::TurnCounter()
{
  m_turn = 0;
}

UAL::QT::TurnCounter* UAL::QT::TurnCounter::getInstance()
{
  if(s_theInstance == 0) {
    s_theInstance = new UAL::QT::TurnCounter();
  }
  return s_theInstance;
}
