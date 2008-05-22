// Library     : PAC
// File        : Optics/PacVTps.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Optics/PacVTps.h"

PacVTps::PacVTps()
{
}

PacVTps::PacVTps(const ZLIB::VTps& vtps)
{ 
  create(vtps); 
}  

PacVTps::PacVTps(const PacVTps& map)
{
  create(map);
}

PacVTps& PacVTps::operator=(const ZLIB::VTps& vtps) 
{ 
  create(vtps); 
  return *this;
}

PacVTps& PacVTps::operator=(const PacVTps& map)
{
  create(map); 
  return *this;
}

PacVTps&  PacVTps::operator*=(const PacVTps& rhs)
{
  if(counter->pointee != 0 && rhs.counter->pointee != 0){
    *(counter->pointee) *= *(rhs.counter->pointee);
  }
  return *this;
}

// Access operators

int PacVTps::size() const
{
  return counter->pointee != 0 ? counter->pointee->size() : 0;
}

unsigned int PacVTps::order() const
{
  return counter->pointee != 0 ? counter->pointee->order() : 0;
}


void PacVTps::order(unsigned int o)
{
  if(counter->pointee) counter->pointee->order(o);
}


int PacVTps::mltOrder() const
{
  return counter->pointee != 0 ? counter->pointee->mltOrder() : 0;
}

void PacVTps::mltOrder(int order)
{
  if(counter->pointee) { counter->pointee->mltOrder(order); }
}

double PacVTps::operator()(int i, int j) const
{
  return counter->pointee != 0 ? counter->pointee->vtps(i, j) : 0;
}

double& PacVTps::operator()(int i, int j) 
{
  if(counter->pointee == 0) {
    string msg = "Error : PacVTps::operator(int d, int index) map is not defined\n";
    PacDomainError(msg).raise();
  }
  return counter->pointee->vtps(i, j);
}

// I/O methods

void PacVTps::read(const char* nfile)
{
  ZLIB::VTps vtps; 
  vtps.read(nfile); 
  create(vtps); 
}

void PacVTps::write(const char* nfile)
{
  if(count()){
    ZLIB::VTps* pointee = counter->pointee;
    pointee->write(nfile);
  } 
  else{
    ZLIB::VTps vtps; 
    vtps.write(nfile);
  }
}



