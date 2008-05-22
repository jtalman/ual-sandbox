// Library     : Teapot
// File        : Main/TeapotMapService.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_MAP_SERVICE_H
#define TEAPOT_MAP_SERVICE_H

#include "PAC/Beam/Bunch.hh"
#include "Optics/PacVTps.h"
#include "Math/TeapotEigenBasis.h"


class Teapot;

class TeapotMapService
{
 public:

  // Constructor

  TeapotMapService(Teapot& code);

  // Find the one-turn map

  void define(/*out*/ PacVTps& map, 
	      /*in*/ const PAC::BeamAttributes& att, 
	      /*in*/ int order) const ;

  // Propagate the map from the element (1) to the element (2)

  void propagate(/*out*/ PacVTps& map, 
		 /*in*/ PAC::BeamAttributes& att, 
		 /*in*/ int index1, 
		 /*in*/ int index2) const;

  // Transform the one-turn map into the eigenbasis

  void transformOneTurnMap(/*out*/ PacVTps& output,
			   /*in*/ const PacVTps& oneTurn) const;

 // Transform the sector map into the eigenbasis

  void transformSectorMap(/*out*/ PacVTps& output, 
			  /*inout*/ PacVTps& oneTurn,
			  /*in*/ const PacVTps& sector) const;

 private:

  Teapot& code_;

  // Copy TeapotMatrix into PacTMap
  void matrix2map(/*in*/ const TeapotMatrix& matrix, 
		  /*out*/ PacVTps& map) const; 
 
  // Copy PacTMap into TeapotMatrix 
  void map2matrix(/*in*/ const PacVTps& map,
		  /*out*/ TeapotMatrix& matrix) const;
  
  // Find the original circumference
  double getLength(const PAC::BeamAttributes& ba) const;
  
};

#endif

