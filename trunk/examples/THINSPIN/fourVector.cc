// Program     :
// File        :
// Copyright   : see Copyright file
// Author      : 

#include "fourVector.hh"

// Constructor

THINSPIN::fourVector::fourVector() : m_data(4, 0.0)
{
}

// Copy onstructor
THINSPIN::fourVector::fourVector(const THINSPIN::fourVector& p) {
  m_data = p.m_data;
}

// Destructor

THINSPIN::fourVector::~fourVector(){
}

// Copy operator

const THINSPIN::fourVector& THINSPIN::fourVector::operator=(const THINSPIN::fourVector& p){
  if(this == &p) return *this;
  m_data = p.m_data;
  return *this;
}

// Access 

void THINSPIN::fourVector::set(double v0, double v1, double v2, double v3)
{
  set0(v0); 
  set1(v1); 
  set2(v2); 
  set3(v3); 
}

double  THINSPIN::fourVector::get0() const { return m_data[0]; }
void    THINSPIN::fourVector::set0(double v) { m_data[0] = v; }

double  THINSPIN::fourVector::get1() const { return m_data[1]; }
void    THINSPIN::fourVector::set1(double v) { m_data[1] = v; }

double  THINSPIN::fourVector::get2() const { return m_data[2]; }
void    THINSPIN::fourVector::set2(double v) { m_data[2] = v; }

double  THINSPIN::fourVector::get3() const { return m_data[3]; }
void    THINSPIN::fourVector::set3(double v) { m_data[3] = v; }

double&  THINSPIN::fourVector::operator[](int index) { return m_data[index]; }
double   THINSPIN::fourVector::operator[](int index) const { return m_data[index]; }

void     THINSPIN::fourVector::setCoordinate(int index, double v) { m_data[index] = v; }

int      THINSPIN::fourVector::size() const { return m_data.size(); }


// Assignment operators

const THINSPIN::fourVector& THINSPIN::fourVector::operator+=(const THINSPIN::fourVector& p)
{
  for(unsigned int i=0; i < m_data.size(); i++) m_data[i] += p.m_data[i];
  return *this;
}

const THINSPIN::fourVector& THINSPIN::fourVector::operator-=(const THINSPIN::fourVector& p)
{
  for(unsigned int i=0; i < m_data.size(); i++) m_data[i] -= p.m_data[i];
  return *this;
}

const THINSPIN::fourVector& THINSPIN::fourVector::operator*=(double v)
{
  for(unsigned int i=0; i < m_data.size(); i++) m_data[i] *= v;
  return *this;
}

const THINSPIN::fourVector& THINSPIN::fourVector::operator/=(double v)
{
  if(v == 0.0){
    std::string msg = "Error : THINSPIN::fourVector::operator/=(double v) : v == 0.0 \n";
//  PacDomainError(msg).raise();
  }
  return operator*=(1./v);
}

THINSPIN::fourVector THINSPIN::fourVector::operator+(const THINSPIN::fourVector& p)
{
  THINSPIN::fourVector tmp(*this);
  tmp += p;
  return tmp;
}

THINSPIN::fourVector THINSPIN::fourVector::operator-(const THINSPIN::fourVector& p)
{
  THINSPIN::fourVector tmp(*this);
  tmp -= p;
  return tmp;
}

THINSPIN::fourVector THINSPIN::fourVector::operator*(double v)
{
  THINSPIN::fourVector tmp(*this);
  tmp *= v;
  return tmp;
}

THINSPIN::fourVector THINSPIN::fourVector::operator/(double v)
{
  THINSPIN::fourVector tmp(*this);
  tmp /= v;
  return tmp;
}
