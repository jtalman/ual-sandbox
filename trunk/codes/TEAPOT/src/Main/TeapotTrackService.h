// Library     : Teapot
// File        : Main/TeapotTrackService.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_TRACK_SERVICE_H
#define TEAPOT_TRACK_SERVICE_H

#include "PAC/Beam/Bunch.hh"


class Teapot;

class TeapotTrackService
{
 public:

  // Constructor

  TeapotTrackService(Teapot& code);

  // Propagate a bunch of particles
       
  void propagate(PAC::Bunch& bunch, int turns = 1);
  void propagate(PAC::Bunch& bunch, int index1, int index2);


 protected:

  Teapot& code_;

  double getLength(const PAC::BeamAttributes& ba);

};

#endif
