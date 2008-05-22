// Library     : Teapot
// File        : Main/TeapotMapService.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Integrator/TeapotDAIntegrator.h"
#include "Main/TeapotMapService.h"
#include "Main/Teapot.h"

// Constructor
TeapotMapService::TeapotMapService(Teapot& code)
  : code_(code)
{
}

// Find an one-turn map
void TeapotMapService::define(/*out*/ PacVTps& vtps, 
			      /*in*/ const PAC::BeamAttributes& beam, 
			      /*in*/ int order) const
{  

  int mltOrder = vtps.mltOrder();
  vtps.mltOrder(order);

  ZLIB::VTps map(TEAPOT_DIMENSION); map = 1.;


  for(int i=0; i < TEAPOT_DIMENSION; i++) {
    map.vtps(i, 0) = vtps(i, 0); }

  ZLIB::VTps tmp(map);


  PAC::BeamAttributes att = beam;
  double e = att.getEnergy(), m = att.getMass();
  double v0byc = sqrt(e*e - m*m)/e;
  // double total_l = getLength(att);
  // att.revfreq(v0byc*BEAM_CLIGHT/total_l);

  TeapotDAIntegrator integrator;  

  integrator.makeVelocity(map, tmp, v0byc);
  integrator.makeRV(att, map, tmp);

  for(int j = 0; j < code_._nelem; j ++) 
    integrator.propagate(code_._telements[j], map, tmp, att, &v0byc); 


  map.order(map.order());

  vtps = map;
  vtps.mltOrder(mltOrder);

}

// Propagate the map from the element (1) to the element (2)
void TeapotMapService::propagate(/*out*/ PacVTps& vtps, 
				 /*in*/ PAC::BeamAttributes& att, 
				 /*in*/ int index1, 
				 /*in*/ int index2) const
{ 
  ZLIB::VTps* pointee = vtps.operator->();
  if(!pointee) return;

  ZLIB::VTps map(*pointee);
  ZLIB::VTps tmp(map);

  double e = att.getEnergy(), m = att.getMass();
  double v0byc = sqrt(e*e - m*m)/e;
  // double total_l = getLength(att);
  // att.revfreq(v0byc*BEAM_CLIGHT/total_l);

  TeapotDAIntegrator integrator;  

  integrator.makeVelocity(map, tmp, v0byc);
  integrator.makeRV(att, map, tmp);

  for(int j = index1; j < index2; j ++) 
    integrator.propagate(code_._telements[j], map, tmp, att, &v0byc); 

  map.order(map.order());
  vtps = map;

}

void TeapotMapService::transformOneTurnMap(/*out*/ PacVTps& output,
					   /*in*/ const PacVTps& oneTurn) const
{
  int i, j, size = min(oneTurn.size(), 4);

  // Write the map into the Teapot matrix

  TeapotMatrix M(size, size);
  map2matrix(oneTurn, M);

  // Make the eigen basis of M matrix

  TeapotEigenBasis eigenBasis(M);

  if(!eigenBasis.isValid()) {
    matrix2map(M, output);
    return;
  }

  // Define the G matrix, an array of eigen vectors

  TeapotMatrix G = eigenBasis.eigenVectors(); 

  // Transpose the G matrix

  TeapotMatrix GT(size, size);

  for(i = 0; i < size; i++)
    for(j = 0; j < size; j++)
      GT[i][j] = G[j][i];    

  // Inverse the GT matrix

  TeapotMatrix GTinv = GT.inverse();

  // Transform the transfer matrix

  M = GTinv*M*GT;

  matrix2map(M, output);
}

void TeapotMapService::transformSectorMap(/*out*/ PacVTps& output,
					  /*inout*/ PacVTps& oneTurn,
					  /*in*/ const PacVTps& sector) const
{  
  int i, j, size = min(oneTurn.size(), 4);

  // Write oneTurnMap and sector mapsinto the Teapot matricies

  TeapotMatrix M0(size, size), M01(size, size);

  map2matrix(oneTurn, M0);
  map2matrix(sector,  M01);

  // Produce symplectic conjugation

  TeapotMatrix Mbar01 = M01.symplecticConjugation();

  // Find the one-turn matrix at (1)

  TeapotMatrix M1 = M01*M0*Mbar01;


  // Find G0 and G1

  TeapotEigenBasis eigenBasis0(M0), eigenBasis1(M1) ;  

  if(eigenBasis0.isValid() == 0 ||eigenBasis1.isValid() == 0) {
    matrix2map(M01, output);
    return;
  }

  TeapotMatrix G0 = eigenBasis0.eigenVectors(); 
  TeapotMatrix G1 = eigenBasis1.eigenVectors(); 

  // Transpose the G0 and G1 matricies

  TeapotMatrix GT0(size, size), GT1(size, size);

  for(i = 0; i < size; i++)
    for(j = 0; j < size; j++){
      GT0[i][j] = G0[j][i];
      GT1[i][j] = G1[j][i];  
    } 

  // Inverse the GT1 matrix

  TeapotMatrix GT1inv = GT1.inverse();

  // Transform the transfer matrix

  TeapotMatrix Mout = GT1inv*M01*GT0;

  matrix2map(M1, oneTurn);
  matrix2map(Mout, output);  

}

// Copy TeapotMatrix into PacTMap
void TeapotMapService::matrix2map(/*in*/ const TeapotMatrix& matrix, 
				  /*out*/ PacVTps& map) const
{

  map.order(1);

  int i, j, size = matrix.rows();
  for(i = 0; i < size; i++){
    map(i, 0) = 0.0;
    for(j = 0; j < size; j++){
      map(i, j + 1) = matrix[i][j]; 
    }
  }   
}	

// Copy PacTMap into TeapotMatrix			   
void TeapotMapService::map2matrix(/*in*/ const PacVTps& map,
				  /*out*/ TeapotMatrix& matrix) const
{
  int i, j, size = matrix.rows();
  for(i = 0; i < size; i++)
    for(j = 0; j < size; j++)
     matrix[i][j]  = map(i, j + 1); 
}



// Find the original circumference
double TeapotMapService::getLength(const PAC::BeamAttributes&) const
{
  PacSurveyData survey;
  code_.survey(survey);
  double l = survey.survey().suml();

  return l;
}



