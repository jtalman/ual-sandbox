// Program     : PAC
// File        : PAC/Beam/Particle.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "PAC/Beam/Particle.hh"

PAC::Spin PAC::Particle::s_spin;

PAC::Particle::Particle() 
  : m_id(0), m_flag(0)
{
  m_spin = 0;
}

PAC::Particle::Particle(const Particle& p) 
  : m_flag(0)
{
  m_spin = 0;
  initialize(p);
}

PAC::Particle::~Particle()
{
  deleteSpin();
}

const PAC::Particle& PAC::Particle::operator = (const PAC::Particle& p)
{
  initialize(p);
  return *this;
}

bool PAC::Particle::isLost() const
{
  return m_flag > 0;
}

int PAC::Particle::getFlag() const
{
  return m_flag;
}

void PAC::Particle::setFlag(int flag)
{
  m_flag = flag;
}

PAC::Position& PAC::Particle::getPosition()
{
  return m_position;
}

const PAC::Position& PAC::Particle::getPosition() const
{
  return m_position;
}

void PAC::Particle::setPosition(const Position& position)
{
  m_position = position;
}

PAC::Spin* PAC::Particle::getSpin()
{
  return m_spin;
}

const PAC::Spin& PAC::Particle::getSpin() const
{
  if(m_spin != 0) return *m_spin;
  return s_spin;
}

void PAC::Particle::setSpin(const Spin& spin)
{
  if(m_spin == 0) m_spin = new PAC::Spin();
  (*m_spin) = spin;
}

void PAC::Particle::initialize(const PAC::Particle& p)
{
  m_id = p.m_id;

  setFlag(p.m_flag); 
  setPosition(p.m_position);

  if(p.m_spin != 0) setSpin(*p.m_spin);
  else deleteSpin();
}

void PAC::Particle::deleteSpin()
{
  if(m_spin != 0) delete m_spin;
  m_spin = 0;
}
