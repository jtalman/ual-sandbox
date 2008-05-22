// Library     : Teapot
// File        : Main/TeapotClOrbitService.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_CLORBIT_SERVICE
#define TEAPOT_CLORBIT_SERVICE

#include "PAC/Beam/Bunch.hh"
#include "Optics/PacTMap.h"
#include "Optics/PacTwissData.h"
#include "SMF/PacElemMultipole.h"

#include "Math/TeapotMatrix.h"

class Teapot;

class TeapotClOrbitService
{
 public:

  // Constructor

  TeapotClOrbitService(Teapot& code);
  virtual ~TeapotClOrbitService();

  // Methods

  // Find a closed orbit
  void define(PAC::Position& orbit, const PAC::BeamAttributes& att);
  void define(PAC::Position& orbit, const PAC::BeamAttributes& att, 
	      PAC::Position* ps, const PacVector<int>& dets);

  // Propagate a closed orbit
  int propagate(PAC::Position& orbit, const PAC::BeamAttributes& att, int index1, int index2);

  // Steer a closed orbit by some method
  void steer(PAC::Position& orbit, const PAC::BeamAttributes& att,
	     const PacVector<int>& ads, const PacVector<int>& dets, int method, char plane); 

 // Steer by Sliding Bumps method
 void steerSB(PAC::Position& orbit, const PAC::BeamAttributes& att,
	     const PacVector<int>& ads, const PacVector<int>& dets, char plane); 
// Steer by Least Square method
 void steerLS(PAC::Position& orbit, const PAC::BeamAttributes& att,
	     const PacVector<int>& ads, const PacVector<int>& dets, char plane);  


 protected:

  void closedOrbit(PAC::Position& p, const PacVTps& map, int order);
  void closedMatrix(TeapotMatrix& matrix, const PacVTps& map);

  void knkmath(const PacTwissData& twiss, char plane);

  void openSets(const PacVector<int>& ads, const PacVector<int>& dets); 
  void addSets(const PacVector<double>& kicks, char plane);
  void closeSets();

 private:

  Teapot& code_;

  // Adjusters

  int aSize_;
  PacTwissData* atwiss_;
  PacElemMultipole **adjusters_;

  // Detectors

  int dSize_;
  PacTwissData* dtwiss_;
  PAC::Position*  detectors_;
  
};

#endif
