// Library     : PAC
// File        : Survey/PacSurvey.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Survey/PacSurvey.h"

// Private methods

void PacSurvey::create()
{
  _size = 7;
  
  data = new double[_size];
  if(!data) {
    cerr << "Error in PacSurvey::create() : allocation failure \n";
    assert(data);
  }
}

void PacSurvey::initialize()
{
  for(int i=0; i < _size; i++) data[i] = 0.0;
}

void PacSurvey::initialize(const PacSurvey& s)
{
  for(int i=0; i < _size; i++) data[i] = s.data[i];
}

void PacSurvey::check(int i) const
{
  if(i < 0 || i >= _size) {
    cerr << "Error in PacSurvey::check(int i) : "
         << "i(" << i << ") < 0 || >= " << _size << "\n";
    assert((i > 0 && i < _size));
  }
}

int PacSurvey::index(const char* id) const
{
  int isize = 6;
  string indexes("x     y     z     suml  theta phi   psi   ");
  int i = indexes.find(id);
  if(i < 0) {
    cerr << "Error in PacSurvey::index(const char* id) : "
         << "id(" << id << ") doesn't match " << indexes << "\n";
    assert(i > 0);
  }

  return i/isize;
}
