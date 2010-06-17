// Program     :
// File        : 
// Copyright   : see Copyright file
// Author      : J. Talman

#include "threeVector.hh"

// Constructor

THINSPIN::threeVector::threeVector() : m_data(3, 0.0)
{
}

// Copy constructor
THINSPIN::threeVector::threeVector(const THINSPIN::threeVector& p) {
  m_data = p.m_data;
}

// Destructor

THINSPIN::threeVector::~threeVector(){
}

// Copy operator

const THINSPIN::threeVector& THINSPIN::threeVector::operator=(const THINSPIN::threeVector& p){
  if(this == &p) return *this;
  m_data = p.m_data;
  return *this;
}

// Access 

void THINSPIN::threeVector::set(double xi, double yi, double zi)
{
  setX(xi); 
  setY(yi); 
  setZ(zi);
}

double  THINSPIN::threeVector::getX() const { return m_data[0]; }
void    THINSPIN::threeVector::setX(double v) { m_data[0] = v; }

double  THINSPIN::threeVector::getY() const { return m_data[1]; }
void    THINSPIN::threeVector::setY(double v) { m_data[1] = v; }

double  THINSPIN::threeVector::getZ() const { return m_data[2]; }
void    THINSPIN::threeVector::setZ(double v) { m_data[2] = v; }

double&  THINSPIN::threeVector::operator[](int index) { return m_data[index]; }
double   THINSPIN::threeVector::operator[](int index) const { return m_data[index]; }

void     THINSPIN::threeVector::setCoordinate(int index, double v) { m_data[index] = v; }

int      THINSPIN::threeVector::size() const { return m_data.size(); }


// Assignment operators

const THINSPIN::threeVector& THINSPIN::threeVector::operator+=(const THINSPIN::threeVector& p)
{
  for(unsigned int i=0; i < m_data.size(); i++) m_data[i] += p.m_data[i];
  return *this;
}

const THINSPIN::threeVector& THINSPIN::threeVector::operator-=(const THINSPIN::threeVector& p)
{
  for(unsigned int i=0; i < m_data.size(); i++) m_data[i] -= p.m_data[i];
  return *this;
}

const THINSPIN::threeVector& THINSPIN::threeVector::operator*=(double v)
{
  for(unsigned int i=0; i < m_data.size(); i++) m_data[i] *= v;
  return *this;
}

const THINSPIN::threeVector& THINSPIN::threeVector::operator/=(double v)
{
  if(v == 0.0){
    std::string msg = "Error : THINSPIN::threeVector::operator/=(double v) : v == 0.0 \n";
//  PacDomainError(msg).raise();
  }
  return operator*=(1./v);
}

THINSPIN::threeVector THINSPIN::threeVector::operator+(const THINSPIN::threeVector& p)
{
  THINSPIN::threeVector tmp(*this);
  tmp += p;
  return tmp;
}

THINSPIN::threeVector THINSPIN::threeVector::operator-(const THINSPIN::threeVector& p)
{
  THINSPIN::threeVector tmp(*this);
  tmp -= p;
  return tmp;
}

THINSPIN::threeVector THINSPIN::threeVector::operator*(double v)
{
  THINSPIN::threeVector tmp(*this);
  tmp *= v;
  return tmp;
}

THINSPIN::threeVector THINSPIN::threeVector::operator/(double v)
{
  THINSPIN::threeVector tmp(*this);
  tmp /= v;
  return tmp;
}
