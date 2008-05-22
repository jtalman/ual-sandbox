// Library     : Teapot
// File        : Main/TeapotChromService.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_CHROM_SERVICE
#define TEAPOT_CHROM_SERVICE

#include "PAC/Beam/Bunch.hh"
#include "Optics/PacVTps.h"
#include "Optics/PacChromData.h"
#include "SMF/PacElemMultipole.h"

class Teapot;

class TeapotChromService
{
 public:

  // Constructor

  TeapotChromService(Teapot& code);
  virtual ~TeapotChromService();    

  // Methods

  // Find initial Chrom parameters

  void define(PacChromData& chrom, const PAC::BeamAttributes& beam, const PAC::Position& orbit);

  // Fit Chrom parameters

  void add(const PAC::BeamAttributes& beam, const PAC::Position& orbit,
	   const PacVector<int>& bf, const PacVector<int>& bd, 
	   double chromx, double chromy, 
	   int numtries, double tolerance, double stepsize);

  void multiply(const PAC::BeamAttributes& beam, const PAC::Position& orbit,
		const PacVector<int>& bf, const PacVector<int>& bd, 
		double chromx, double chromy, 
		int numtries, double tolerance, double stepsize);

 protected:

  void findDR(TeapotMatrix& dr, const PacTMap& map, const PAC::Position& d);
		
  void   openMlt(const PacVector<int>& ifs, const PacVector<int>& ids);
  double multiplyBfs(double b2f);
  double multiplyBds(double b2d);
  double getBf();
  double getBd();
  void   closeMlt();

 private:

  Teapot& code_;

  int bfSize_, bdSize_;
  PacElemMultipole **bfs_, **bds_;
  
};

#endif
