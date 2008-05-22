// Program     : PAC
// File        : PAC/Beam/Position.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "PAC/Common/PacException.h"
#include "PAC/Beam/Position.hh"

// Constructor

PAC::Position::Position() : m_data(6, 0.0)
{
}

// Copy onstructor
PAC::Position::Position(const PAC::Position& p) {
  m_data = p.m_data;
}

// Destructor

PAC::Position::~Position(){
}

// Copy operator

const PAC::Position& PAC::Position::operator=(const PAC::Position& p){
  if(this == &p) return *this;
  m_data = p.m_data;
  return *this;
}

// Access 

void PAC::Position::set(double xi, double pxi, double yi, double pyi, double cti, double dei)
{
  setX(xi); 
  setPX(pxi); 
  setY(yi); 
  setPY(pyi); 
  setCT(cti); 
  setDE(dei);
}

double  PAC::Position::getX() const { return m_data[0]; }
void    PAC::Position::setX(double v) { m_data[0] = v; }

double  PAC::Position::getPX() const { return m_data[1]; }
void    PAC::Position::setPX(double v) { m_data[1] = v; }

double  PAC::Position::getY() const { return m_data[2]; }
void    PAC::Position::setY(double v) { m_data[2] = v; }

double  PAC::Position::getPY() const { return m_data[3]; }
void    PAC::Position::setPY(double v) { m_data[3] = v; }

double  PAC::Position::getCT() const { return m_data[4]; }
void    PAC::Position::setCT(double v) { m_data[4] = v; }

double  PAC::Position::getDE() const { return m_data[5]; }
void    PAC::Position::setDE(double v) { m_data[5] = v; }

double&  PAC::Position::operator[](int index) { return m_data[index]; }
double   PAC::Position::operator[](int index) const { return m_data[index]; }

void     PAC::Position::setCoordinate(int index, double v) { m_data[index] = v; }

int      PAC::Position::size() const { return m_data.size(); }


// Assignment operators

const PAC::Position& PAC::Position::operator+=(const PAC::Position& p)
{
  for(unsigned int i=0; i < m_data.size(); i++) m_data[i] += p.m_data[i];
  return *this;
}

const PAC::Position& PAC::Position::operator-=(const PAC::Position& p)
{
  for(unsigned int i=0; i < m_data.size(); i++) m_data[i] -= p.m_data[i];
  return *this;
}

const PAC::Position& PAC::Position::operator*=(double v)
{
  for(unsigned int i=0; i < m_data.size(); i++) m_data[i] *= v;
  return *this;
}

const PAC::Position& PAC::Position::operator/=(double v)
{
  if(v == 0.0){
    std::string msg = "Error : PAC::Position::operator/=(double v) : v == 0.0 \n";
    PacDomainError(msg).raise();
  }
  return operator*=(1./v);
}

PAC::Position PAC::Position::operator+(const PAC::Position& p)
{
  PAC::Position tmp(*this);
  tmp += p;
  return tmp;
}

PAC::Position PAC::Position::operator-(const PAC::Position& p)
{
  PAC::Position tmp(*this);
  tmp -= p;
  return tmp;
}

PAC::Position PAC::Position::operator*(double v)
{
  PAC::Position tmp(*this);
  tmp *= v;
  return tmp;
}

PAC::Position PAC::Position::operator/(double v)
{
  PAC::Position tmp(*this);
  tmp /= v;
  return tmp;
}

/*
PAC::Position operator-(const PAC::Position& p)
{
  PAC::Position tmp(p);
  tmp *= -1.;
  return tmp;
}

PAC::Position operator*(double v, const PAC::Position& p)
{
  PAC::Position tmp(p);
  tmp *= v;
  return tmp;
}
*/






