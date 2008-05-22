// Library       : SIMBAD
// File          : SIMBAD/SC/LFactors.cc
// Copyright     : see Copyright file
// Author        : N.Malitsky 

#include "SIMBAD/SC/LFactors.hh"

SIMBAD::LFactors* SIMBAD::LFactors::s_theInstance = 0;

// Constructor

SIMBAD::LFactors::LFactors(int maxBunchSize)
  :m_lFactors(maxBunchSize, 1.0)
{
}

// Get singleton

SIMBAD::LFactors& SIMBAD::LFactors::getInstance(int maxBunchSize)
{
  if(!s_theInstance){
    s_theInstance = new SIMBAD::LFactors(maxBunchSize);
  }
  return *s_theInstance;
}

int SIMBAD::LFactors::getSize() const
{
  return m_lFactors.size();
}

double SIMBAD::LFactors::getElement(int index) const
{
  return m_lFactors[index];
}

void SIMBAD::LFactors::setElement(double value, int index)
{
  m_lFactors[index] = value;
}
