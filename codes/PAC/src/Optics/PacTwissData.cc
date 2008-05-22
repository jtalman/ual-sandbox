// Program     : PAC
// File        : Beam/PacTwissData.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Optics/PacTwissData.h"

// Constructors & destructor

PacTwissData::PacTwissData(int dim)
  : data_(dim, SIZE)
{
}

PacTwissData::PacTwissData(const PacTwissData& right)
  : data_(right.data_)
{
}

PacTwissData::~PacTwissData()
{
}

// Operators

PacTwissData& PacTwissData::operator = (const PacTwissData& right)
{
  data_ = right.data_;
  return *this;
}

// Access methods

int PacTwissData::dimension() const
{
  return data_.rows();
}

double PacTwissData::value(int d, int index) const
{
  return d >= data_.rows() ? 0 : data_[d][index];
}

double& PacTwissData::value(int d, int index)
{
  if(d >= data_.rows() ){
    string msg = "Error : PacTwissData::value(int d, int index) d > dimension \n";
    PacDomainError(msg).raise();
  }
  return data_[d][index];
}

