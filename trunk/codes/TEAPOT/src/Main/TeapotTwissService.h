// Library     : Teapot
// File        : Main/TeapotTwissService.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_TWISS_SERVICE
#define TEAPOT_TWISS_SERVICE

#include "PAC/Beam/Bunch.hh"
#include "Optics/PacVTps.h"
#include "Optics/PacTwissData.h"
#include "SMF/PacElemMultipole.h"

#include "Math/TeapotMatrix.h"

class Teapot;

class TeapotTwissService
{
 public:

  // Constructor

  TeapotTwissService(Teapot& code);
  virtual ~TeapotTwissService();

  // Methods

  // Find initial Twiss parameters

  void define(PacTwissData& twiss, const PacVTps& map);
  void define(PacTwissData& twiss, const PAC::BeamAttributes& att, const PAC::Position& orbit);
  PacTwissData define(PacTwissData* twiss, const PacVector<int> indices, 
	      const PAC::BeamAttributes& att, const PAC::Position& orbit); 

  // Propagate Twiss parameters

  void propagate(PacTwissData& twiss, const PacVTps& map);

  // Fit Twiss parameters

  void add(const PAC::BeamAttributes& att, const PAC::Position& orbit,
	   const PacVector<int>& b1f, const PacVector<int>& b1d, 
	   double mux, double muy,
	   int numtries, double tolerance, double stepsize);

  void multiply(const PAC::BeamAttributes& att, const PAC::Position& orbit,
		const PacVector<int>& b1f, const PacVector<int>& b1d, 
		double mux, double muy,
		int numtries, double tolerance, double stepsize);

 protected:

  void closedEta(PAC::Position& eta, const PacVTps& map);
  void closedMatrix(TeapotMatrix& matrix, const PacVTps& map);

  void   openMlt(const PacVector<int>& ifs, const PacVector<int>& ids);
  double multiplyBfs(double b1);
  double multiplyBds(double b1);
  double getBf();
  double getBd();
  void   closeMlt();

  
 private:

  Teapot& code_;

  int bfSize_, bdSize_;
  PacElemMultipole **bfs_, **bds_;

};

#endif


