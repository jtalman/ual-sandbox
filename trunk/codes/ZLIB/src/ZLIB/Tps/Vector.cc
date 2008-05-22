// Library     : ZLIB
// File        : ZLIB/Tps/Vector.cc 
// Copyright   : see Copyright file
// Author      : Yiton Yan
// C++ version : Nikolay Malitsky 

#include "ZLIB/Tps/Vector.hh"

ZLIB::Vector::Vector(unsigned int s, double value)
  : size_(0)
{
  vector_ = 0;
  if(s < 1) return;
  
  vector_ = new double[s];
  if(!vector_){
    cerr << "Error: ZLIB::Vector::Vector(unsigned int s, double value) : allocation failure \n";
    assert(vector_);
  }

  size_ = s;
  for(unsigned int i = 0; i < size_; i++) vector_[i] = value;

}

ZLIB::Vector::Vector(const ZLIB::Vector& rhs)
  : size_(0)
{
  vector_ = 0;
  initialize(rhs);
}

ZLIB::Vector::~Vector()
{
  erase();
}

ZLIB::Vector& ZLIB::Vector::operator=(const ZLIB::Vector& rhs)
{
  if(this != &rhs){
    erase();
    initialize(rhs);
  }
  return *this;
}

void ZLIB::Vector::initialize(const ZLIB::Vector& rhs)
{
  vector_ = new double[rhs.size_];
  if(!vector_){
    cerr << "Error: ZLIB::Vector::initialize(const ZLIB::Vector& rhs) : allocation failure \n";
    assert(vector_);
  }

  size_ = rhs.size_;
  for(unsigned int i = 0; i < size_; i++) vector_[i] = rhs.vector_[i];

}

void ZLIB::Vector::erase()
{
  delete [] vector_;
  vector_ = 0;
  size_ = 0;
}
