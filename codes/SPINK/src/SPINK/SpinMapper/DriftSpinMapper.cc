// Library       : SPINK
// File          : SPINK/SpinMapper/DriftSpinMapper.cc
// Copyright     : see Copyright file
// Author        : A.Luccio
// C++ version   : N.Malitsky 

#include "PAC/Beam/Bunch.hh"
#include "SPINK/SpinMapper/DriftSpinMapper.hh"

SPINK::DriftSpinMapper::DriftSpinMapper()
{
}

SPINK::DriftSpinMapper::DriftSpinMapper(const SPINK::DriftSpinMapper& sm)
{
  copy(sm);
}

SPINK::DriftSpinMapper::~DriftSpinMapper()
{
}

UAL::PropagatorNode* SPINK::DriftSpinMapper::clone()
{
  return new SPINK::DriftSpinMapper(*this);
}

void SPINK::DriftSpinMapper::propagate(UAL::Probe& b)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(b);
  m_tracker->propagate(bunch);  
}

void SPINK::DriftSpinMapper::copy(const SPINK::DriftSpinMapper&)
{
}
