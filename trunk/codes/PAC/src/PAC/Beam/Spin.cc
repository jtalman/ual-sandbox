// Program     : PAC
// File        : PAC/Beam/Spin.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "PAC/Beam/Spin.hh"

// Constructor

PAC::Spin::Spin()
{
  m_sx = 0.0;
  m_sy = 1.0;
  m_sz = 0.0;
}

PAC::Spin::Spin(double sx, double sy, double sz)
{
  m_sx = sx;
  m_sy = sy;
  m_sz = sz;
}


// Copy onstructor
PAC::Spin::Spin(const PAC::Spin& s) {
  m_sx = s.m_sx;
  m_sy = s.m_sy;
  m_sz = s.m_sz;
}

// Destructor

PAC::Spin::~Spin(){
}

// Copy operator

const PAC::Spin& PAC::Spin::operator=(const PAC::Spin& s){
  if(this == &s) return *this;
  set(s.m_sx, s.m_sy, s.m_sz);
  return *this;
}

// Access 

void PAC::Spin::set(double sx, double sy, double sz)
{
  m_sx = sx;
  m_sy = sy;
  m_sz = sz;
}

double  PAC::Spin::getSX() const { return m_sx; }
void    PAC::Spin::setSX(double v) { m_sx = v; }

double  PAC::Spin::getSY() const { return m_sy; }
void    PAC::Spin::setSY(double v) { m_sy = v; }

double  PAC::Spin::getSZ() const { return m_sz; }
void    PAC::Spin::setSZ(double v) { m_sz = v; }

double  PAC::Spin::operator[] (int index) const { 
  switch (index) {
  case 0:
    return m_sx;
    break;
  case 1:
    return m_sy;
    break;
  case 2:
    return m_sz;
    break;
  default:
    return 0.0;
  }
}

int  PAC::Spin::size() const { 
  return 3; 
}



