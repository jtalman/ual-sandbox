// Program     : PAC
// File        : Beam/PacChromData.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Optics/PacChromData.h"

// Constructors & destructor

PacChromData::PacChromData(int dim)
  : twiss_(dim), data_(dim, SIZE)
{
}

PacChromData::PacChromData(const PacChromData& right)
  : twiss_(right.twiss_), data_(right.data_)
{
}

PacChromData::~PacChromData()
{
}

// Operators

PacChromData& PacChromData::operator = (const PacChromData& right)
{
  twiss_ = right.twiss_;
  data_  = right.data_;
  return *this;
}

// Access methods

int PacChromData::dimension() const
{
  return data_.rows();
}

double PacChromData::value(int d, int index) const
{
  return d >= data_.rows() ? 0 : data_[d][index];
}

double& PacChromData::value(int d, int index)
{
  if(d >= data_.rows() ){
    string msg = "Error : PacChromData::value(int d, int index) d > dimension \n";
    PacDomainError(msg).raise();
  }
  return data_[d][index];
}

