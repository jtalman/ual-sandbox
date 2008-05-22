// Library     : Teapot
// File        : Main/TeapotMatrixService.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_MATRIX_SERVICE_H
#define TEAPOT_MATRIX_SERVICE_H

#include "PAC/Beam/Bunch.hh"
#include "Optics/PacTMap.h"
#include "Math/TeapotMatrix.h"
#include "Integrator/TeapotIntegrator.h"

enum TeapotMatrixService_Enum { TMS_PLUS, TMS_MINUS };

class Teapot;

class TeapotMatrixService
{
 public:

  // Constructor

  TeapotMatrixService(Teapot& code);
  virtual ~TeapotMatrixService();

  // Find an one-turn map
       
  void define(PacVTps& vtps, const PAC::BeamAttributes& att, const PAC::Position& delta, int order);

  // Propagate map

  int start(const PAC::Position& orbit, const PAC::BeamAttributes& att, const PAC::Position& delta);
  int next(PacVTps& vtps, PAC::BeamAttributes& att);
  int stop();

  // Decouple map

  void decouple(const PAC::BeamAttributes& beam, const PAC::Position& orbit, 
		const PacVector<int>& a11s,  const PacVector<int>& a12s, 
		const PacVector<int>& a13s,  const PacVector<int>& a14s,
		const PacVector<int>& bfs,   const PacVector<int>& bds,
		double mux, double muy);

 protected:

  // 

  void genRays(PAC::Bunch& bunch, const PAC::BeamAttributes& att, const PAC::Position& delta, int order);
  void getMatrix(PacVTps& map, const PAC::Bunch& bunch, const PAC::Position& delta, int order);


  // Decouple methods

  void makeDecoupleAdjusters(int ia, const PacVector<int>& indices);
  void makeDecoupleFamilies(int ia, const PacVector<int>& index1,
		  const PacVector<int>& index2, TeapotMatrixService_Enum sign2);
  void printDecoupleDeltas(const PacVector<double>& deltas);
  void addDecoupleAdjusters(int i, double v);
  void deleteDecoupleAdjusters(int i);
  void deleteDecoupleAdjusters();
  double normDecoupleAdjusters();

  TeapotVector makeDecoupleValues(double mux, double muy);
  TeapotVector makeDecoupleStep();
  void getDecoupleValues(TeapotVector& values, const PAC::BeamAttributes& beam, const PAC::Position& orbit);
  void printDecoupleValues(const TeapotVector& values);
  

 private:

  Teapot& code_;


  TeapotIntegrator integrator_;
  int              index_;
  double           v0byc_;
  PAC::Position      delta_;
  PAC::Bunch         linRays_;
  PAC::Bunch         tmpRays_;

  // Decouple adjusters

  int* aSizes_;
  double** aSigns_;
  PacElemMultipole*** adjusters_;

  static double pi_;
  

};


#endif
