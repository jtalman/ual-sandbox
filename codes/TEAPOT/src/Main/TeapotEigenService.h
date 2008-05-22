// Library     : Teapot
// File        : Main/TeapotEigenService.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_EIGEN_SERVICE
#define TEAPOT_EIGEN_SERVICE

#include "PAC/Beam/Bunch.hh"
#include "Optics/PacVTps.h"
#include "Optics/PacTwissData.h"
#include "SMF/PacElemMultipole.h"

#include "Math/TeapotEigenBasis.h"

class Teapot;

class TeapotEigenService
{
 public:

  // Constructor
  TeapotEigenService(Teapot& code);

  // Destructor
  virtual ~TeapotEigenService();

  // Find initial Twiss parameters  
  void define(/*out*/ PacTwissData& twiss, 
	      /*in*/ const PacVTps& map) const;
  void propagate(/*out*/ PacTwissData& twiss, 
		 /*in*/ const PacVTps& sector) const;
 protected:

  // Dimension of this service (2)
  inline int dimension() const;

  // Find the closed eta (uncoupled case)
  void closedEta(/*out*/ PAC::Position& eta, 
		 /*in*/ const PacVTps& map) const;

 private:

  Teapot& code_;
      

};

#endif // TEAPOT_EIGEN_SERVICE 
