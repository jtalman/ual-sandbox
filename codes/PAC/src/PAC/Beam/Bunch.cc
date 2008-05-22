// Program     : PAC
// File        : PAC/Beam/Bunch.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include <math.h>
#include "PAC/Common/PacException.h"
#include "PAC/Beam/Bunch.hh"

// Constructors

PAC::Bunch::Bunch(int size) 
  :  m_particles(size), m_size(size) 
{
}

PAC::Bunch::Bunch(const PAC::Bunch& bunch)
  : m_ba(bunch.m_ba), m_particles(bunch.m_particles), m_size(bunch.m_size) 
{
}

PAC::Bunch::~Bunch()
{
}
  
PAC::Bunch& PAC::Bunch::operator=(const PAC::Bunch& bunch)
{
  m_ba        = bunch.m_ba;
  m_particles = bunch.m_particles;
  m_size      = bunch.m_size;

  return *this;
}

void PAC::Bunch::erase(int i)
{
  if(i >= m_size) return;

  m_particles[i] = m_particles[m_size - 1];
  m_size -= 1;
}

PAC::Bunch& PAC::Bunch::add(const PAC::Bunch& bunch)
{
  m_particles.resize(m_size + bunch.size());

  int counter = m_size;
  for(unsigned int i = 0; i < bunch.size(); i++){
    m_particles[counter++] = bunch.m_particles[i];
  }

  m_size +=  bunch.size();

  return *this;
}

void PAC::Bunch::setBeamAttributes(const PAC::BeamAttributes& ba)
{
  m_ba = ba;
}

PAC::BeamAttributes& PAC::Bunch::getBeamAttributes()
{
  return m_ba;
}

const PAC::BeamAttributes& PAC::Bunch::getBeamAttributes() const
{
  return m_ba;
}

int PAC::Bunch::capacity() const 
{
  return m_particles.size();
}

int PAC::Bunch::size() const 
{
  // return m_particles.size();
  return m_size;
}

void PAC::Bunch::resize(int n)
{
  m_particles.resize(n);
  m_size = n;
}

PAC::Particle& PAC::Bunch::getParticle(int index)
{
  return m_particles[index];
} 

void PAC::Bunch::setParticle(int index, const Particle& particle)
{
  m_particles[index] = particle;
}  

PAC::Particle& PAC::Bunch::operator[](int index)
{
  return m_particles[index];
}

const PAC::Particle& PAC::Bunch::operator[](int index) const
{
  return m_particles[index];
}

double PAC::Bunch::moment(int index, int order) const
{
  if(index < 0 || index >= 6){
     std::string msg1   = "Error : PAC::Bunch::moment(int index, int order): ";
     msg1  += "index is out of [0, 6] \n";
 
     PacDomainError(msg1).raise();
  }
  if(order < 0){
     std::string msg1  = "Error : PAC::Bunch::moment(int index, int order): ";
     msg1 += "order < 0 \n";
     PacDomainError(msg1).raise();
  }

  double result = 0;
  for(int i=0; i < size(); i++){
    if(!(*this)[i].getFlag()) result += pow((*this)[i].getPosition()[index], order);
  }
  return result/size();
}

PAC::Position PAC::Bunch::moment(int order) const
{
  PAC::Position result;
  for(int i=0; i < result.size(); i++) result.setCoordinate(i, moment(i, order));
  return result;  
}





