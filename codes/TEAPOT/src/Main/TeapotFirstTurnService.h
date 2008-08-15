// Library     : Teapot
// File        : Main/TeapotFirstTurnService.h
// Copyright   : see Copyright file
// Author      : Vadim Ptitsyn

#ifndef TEAPOT_FIRSTTURN_SERVICE
#define TEAPOT_FIRSTTURN_SERVICE

#include "PAC/Beam/Bunch.hh"
#include "Optics/PacTMap.h"
#include "Optics/PacTwissData.h"
#include "SMF/PacElemMultipole.h"

#include "Math/TeapotMatrix.h"
#include <iostream>

class Teapot;

class TeapotFirstTurnService
{
 public:

  // Constructor

  TeapotFirstTurnService(Teapot& code, double MaxAllowedDev = 1.e-2);
  virtual ~TeapotFirstTurnService();

  // Methods

  // Find an orbit on monitors along the ring until the deviation higher than allowed is met
  int define(PAC::Position& orbit, const PAC::BeamAttributes& att, 
	     PAC::Position* hps, const PacVector<int>& hdets, PAC::Position* vps, const PacVector<int>& vdets,
             int& max1, int& max2, char& max_pln );

  // Propagate an  orbit
  int propagate(PAC::Position& orbit, const PAC::BeamAttributes& att, int index1, int index2);

  // Steer the first turn in both planes
  void steer(PAC::Position& orbit, const PAC::BeamAttributes& att,
	     const PacVector<int>& hads, const PacVector<int>& hdets,
             const PacVector<int>& vads, const PacVector<int>& vdets, 
             const PacTwissData& tw, const int method );


  // Steering methods  
  void steerSB(PAC::Position& orbit, const PAC::BeamAttributes& att,
	     const PacVector<int>& hads, const PacVector<int>& hdets,
             const PacVector<int>& vads, const PacVector<int>& vdets, 
             const PacTwissData& tw);
  void steerGrote(PAC::Position& orbit, const PAC::BeamAttributes& att,
	     const PacVector<int>& hads, const PacVector<int>& hdets,
             const PacVector<int>& vads, const PacVector<int>& vdets, 
             const PacTwissData& tw);

  void PrintDetectors(ostream& out=cout, int plane=0);

 protected:
  
  void FindLimits(int& hdtsmin, int& hdtsmax, int max1);

  void openSets(const PacVector<int>& hads, const PacVector<int>& hdets,
                const PacVector<int>& vads, const PacVector<int>& vdets); 
//  void addSets(const PacVector<double>& kicks, char plane);
  void closeSets();

 private:

  Teapot& code_;
  double MaxAllowedDeviation_;
  int hmaxdets_, vmaxdets_;

  // Adjusters

  int haSize_;
  PacTwissData* hatwiss_;
  PacElemMultipole **hadjusters_;

  int vaSize_;
  PacTwissData* vatwiss_;
  PacElemMultipole **vadjusters_;

  // Detectors

  int hdSize_;
  PacTwissData* hdtwiss_;
  PAC::Position*  hdetectors_;
  int vdSize_;
  PacTwissData* vdtwiss_;
  PAC::Position*  vdetectors_;
};

#endif
