// Program     : Pac
// File        : Optics/PacTMap.cc
// Description : Taylor Map
// Copyright   : see Copyright file
// Authors     : Nikolay Malitsky

#include <stdio.h>
#include <time.h>
#include "Optics/PacTMap.h"

const double APERTURE2 = 1.0;

PAC::Position PacTMap::refOrbit() const
{
  PAC::Position p;

  ZLIB::VTps* pointee = counter->pointee;
  int s = min((int) pointee->size(), p.size());
  for(int i=0; i < s; i++) p.setCoordinate(i,  pointee->vtps(i, 0)); 

  return p;
}

void PacTMap::refOrbit(const PAC::Position& p)
{
  ZLIB::VTps* pointee = counter->pointee;
  int s = min((int) pointee->size(), p.size());

  for(int i=0; i < s; i++) pointee->vtps(i, 0) = p[i];
  for(unsigned int j=s; j < pointee->size(); j++) pointee->vtps(j, 0) = 0.0;  
}

void PacTMap::propagate(PAC::Bunch& bunch, int turns)
{

  for(int i=0; i < turns; i++){
    for(int ip = 0; ip < bunch.size(); ip++){
      if(!bunch[ip].getFlag()){
	propagate(bunch[ip].getPosition());
	check(bunch[ip], ip, i);
      }
    }
  }
}

void PacTMap::check(PAC::Particle& particle, int ip, int turn)
{
  PAC::Position* p = &particle.getPosition();
  double a = p->getX()*p->getX() + p->getY()*p->getY();

  if(a > APERTURE2) {
    cerr << "PacTMap::propagate: particle (" << ip << ") has been lost after " 
	 << turn << " turns \n"; 
    cerr << " x = " << p->getX() << " y = " <<  p->getY() << "\n";
    particle.setFlag(1);
  }
  else              {
    particle.setFlag(0); 
  }
}

void PacTMap::propagate(PAC::Position& p, int turns)
{
  for(int i=0; i < turns; i++) propagate(p);
  return;
}

void PacTMap::propagate(PAC::Position& p)
{
  ZLIB::VTps* pointee = counter->pointee;

  ZLIB::Vector v(p.size());
  for(int i=0; i < p.size(); i++) v[i] = p[i];
  pointee->propagate(v);
  for(int j=0; j < p.size(); j++) p.setCoordinate(j, v[j]);

  return;
}

void PacTMap::propagate(PacVTps& vtps)
{
  vtps *= *this;
}

void PacTMap::propagate(ZLIB::VTps& vtps)
{
   if(counter->pointee != 0) vtps *= *(counter->pointee);
}

void PacTMap::create(int size)
{
  ZLIB::VTps tmp(size);
  tmp = 1;
  PacVTps::create(tmp);
}
