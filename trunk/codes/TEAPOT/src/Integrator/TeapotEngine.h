// Program     : Teapot
// File        : Integrator/TeapotEngine.h
// Copyright   : see Copyright file
// Description : Tracking Engine
// Author      : Nikolay Malitsky

#ifndef TEAPOT_ENGINE_H
#define TEAPOT_ENGINE_H

#include "SMF/PacElemMultipole.h"
#include "SMF/PacElemOffset.h"
#include "SMF/PacElemAperture.h"
#include "SMF/PacElemSolenoid.h"
#include "SMF/PacElemRfCavity.h"
#include "SMF/PacElemRotation.h"

#include "Optics/PacTMap.h"

#include "PAC/Beam/BeamAttributes.hh"
#include "PAC/Beam/Position.hh"

#include "Integrator/TeapotElement.h"

template<class Coordinate, class Position>
class TeapotEngine
{
public:

  // Constructor

  TeapotEngine();

  // Commands

  int propagate(const PacGenElement& ge, PAC::BeamAttributes& ba, Position& p);
  int propagate(const TeapotElement& te, PAC::BeamAttributes& ba, Position& p);

  // Other commands

  int propagate(const TeapotElement& te, 
		Position& p, 
		Position& tmp, 
		PAC::BeamAttributes& ba, 
		double* v0byc);

  virtual void makeVelocity(Position& p, Position& tmp, double v0byc);

  void makeRV(const PAC::BeamAttributes& ba, Position& p, Position& tmp);

 protected:

  // Data

  double _steps[6];
  double _kicks[5];

  enum bucketKey{
    DRIFT       =   1,
    BEND        =   2,
    MULTIPOLE   =   4,
    OFFSET      =   8,
    APERTURE    =  16,
    SOLENOID    =  32,
    RFCAVITY    =  64,
    ROTATION    = 128
  };

  int _pathKey;

  // ElemPart

  PacElemAttributes *_front;
  PacElemAttributes *_end;

  // Element Buckets
  
  PacElemMultipole  *_mult;
  PacElemOffset     *_offset;
  PacElemAperture   *_aperture;
  PacElemSolenoid   *_solenoid;
  PacElemRfCavity   *_rf;
  PacElemRotation   *_rotation;

  // TeapotElement  Data

  double _l;     // length
  int _ir;       // ir
  double _rIr;   // 1./ir

  TeapotElemBend* _bend;

  // ... Element Parts ( Front, Body, End )

  int passFront(const TeapotElement& te,
		Position& p,
		Position& tmp,
		double v0byc);

  int passBody(const TeapotElement& te,
	       Position& p,
	       Position& tmp, 
	       PAC::BeamAttributes& ba,
	       double* v0byc);

  int passEnd(const TeapotElement& te,
	      Position& p,
	      Position& tmp,
	      double v0byc);

  // ... Intervals ( Drift, Dipole, Solenoid )

  void passInterval(int iSlice,
		    int iStep, 
		    const TeapotElement& te, 
		    Position& p,
                    Position& tmp,
                    double v0byc);

  void passDrift(double length, 
		 Position& p, 
		 Position& tmp,
		 double v0byc);

//  void deltaDriftPath(double length, 
//		 Position& p, 
//		 Position& tmp,
//		 double v0byc);

  void passBend( const TeapotElement& te,
		 const TeapotElemSlice& es, 
		 Position& p, 
		 Position& tmp,
		 double v0byc);

  void deltaPath(const TeapotElemSlice& es, 
		 Position& p, 
		 Position& tmp,
		 double v0byc);

  // ... Kicks ( multipole, solenoid, ...)

  void passMltKick(const TeapotElement& te,
		   int iKick,
		   Position& p,
		   Position& tmp,
		   double v0byc);
		       
  virtual void passSlndKick(double length, 
		    Position& p, 
		    Position& tmp,
		    double v0byc);


  virtual void passRfKick(int iKick,
		 Position& p,
                 Position& tmp,
		 PAC::BeamAttributes& ba,
                 double* v0byc) = 0;

  // Aperture	 

  virtual int  testAperture(Position& p) = 0;     

private:

  void initialize();
  void body(const TeapotElement& te);

  int min(int a, int b) { return a<b ? a : b; }
  
};

#include "Integrator/TeapotEngine.icc"

#endif
