// Library     : Teapot
// File        : Main/Teapot.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include <stdio.h>
#include <time.h>

#include "Main/Teapot.h"
#include "Integrator/TeapotIntegrator.h"
#include "Integrator/TeapotDAIntegrator.h"

#include "Main/TeapotMapService.h"
#include "Main/TeapotTrackService.h"
#include "Main/TeapotMatrixService.h"
#include "Main/TeapotClOrbitService.h"
#include "Main/TeapotTwissService.h"
#include "Main/TeapotChromService.h"
#include "Main/TeapotEigenService.h"
#include "Main/TeapotFirstTurnService.h"

// Public methods

// Commands

// Tracking

void Teapot::track(PAC::Bunch& bunch, int turns)
{ 
  TeapotTrackService service(*this);
  service.propagate(bunch, turns);
}

void Teapot::track(PAC::Bunch& bunch, int index1, int index2)
{
  TeapotTrackService service(*this);
  service.propagate(bunch, index1, index2);
}

// Survey

void Teapot::survey(PacSurveyData& survey, int index1, int index2)
{
  for(int j = index1; j < index2; j++) _telements[j].propagate(survey); 
}

// Closed Orbit

void Teapot::clorbit(PAC::Position& orbit, const PAC::BeamAttributes& att)
{
  TeapotClOrbitService service(*this);
  service.define(orbit, att);
}

void Teapot::trackClorbit(PAC::Position& orbit, const PAC::BeamAttributes& att, int i1, int i2)
{
  TeapotClOrbitService service(*this);
  service.propagate(orbit, att, i1, i2);
}

void Teapot::steer(PAC::Position& orbit, const PAC::BeamAttributes& att, 
		   const PacVector<int> ads, const PacVector<int> dets, int method, char plane)
{
  TeapotClOrbitService service(*this);
  service.steer(orbit, att, ads, dets, method, plane);
}

// First Turn

void Teapot::ftsteer(PAC::Position& orbit, const PAC::BeamAttributes& att, 
	     const PacVector<int> hads, const PacVector<int> hdets,
             const PacVector<int> vads, const PacVector<int> vdets,
             double MaxAllowedDev, const PacTwissData& tw, const int method)
{
  TeapotFirstTurnService service(*this, MaxAllowedDev );
  service.steer(orbit, att, hads, hdets, vads, vdets, tw, method); 

}

// Twiss

void Teapot::twiss(PacTwissData& tw, const PAC::BeamAttributes& att, const PAC::Position& orbit)
{
  TeapotTwissService service(*this);
  service.define(tw, att, orbit);
}

void Teapot::twissList(PacTwissData& tw, const PAC::BeamAttributes& att, const PAC::Position& orbit)
{
 
  TeapotTwissService service(*this);
 
  if(_twissList) delete [] _twissList; 
  _twissList = new PacTwissData[size()];
  
  PacVector<int> indices(size());
  int i;
  for(i=0; i< size(); i++){ indices[i] = i;}
  tw = service.define(_twissList, indices, att, orbit);
}

void Teapot::eraseTwissList(){

 if(_twissList) delete [] _twissList;

   _twissList = 0;
}

void Teapot::trackTwiss(PacTwissData& tw, const PacVTps& map)
{
  TeapotTwissService service(*this);
  service.propagate(tw, map);
}

void Teapot::tunethin(const PAC::BeamAttributes& att, const PAC::Position& orbit,
		      const PacVector<int>& bf, const PacVector<int>& bd, 
		      double mux, double muy, char method,
		      int numtries, double tolerance, double stepsize)
{
  TeapotTwissService service(*this);
  switch (method) {
  case '+' :
   service.add(att, orbit, bf, bd, mux, muy, numtries, tolerance, stepsize);
   break;
  case '*' :
    service.multiply(att, orbit, bf, bd, mux, muy, numtries, tolerance, stepsize);
    break;
  default:
    service.multiply(att, orbit, bf, bd, mux, muy, numtries, tolerance, stepsize);
    break;
  }    
}

// Eigen parameters

void Teapot::eigenTwiss(/*out*/ PacTwissData& twiss, 
			/*in*/ const PacVTps& map)
{
  TeapotEigenService service(*this);
  service.define(twiss, map);
}

void Teapot::trackEigenTwiss(/*out*/ PacTwissData& twiss, 
			     /*in*/ const PacVTps& sector)
{
  TeapotEigenService service(*this);
  service.propagate(twiss, sector);
}

// Chromaticity

void Teapot::chrom(PacChromData& ch, const PAC::BeamAttributes& att, const PAC::Position& orbit)
{
  TeapotChromService service(*this);
  service.define(ch, att, orbit);
}

void Teapot::chromfit(const PAC::BeamAttributes& att, const PAC::Position& orbit,
		      const PacVector<int>& bf, const PacVector<int>& bd, 
		      double mux, double muy, char method,
		      int numtries, double tolerance, double stepsize)
{
  TeapotChromService service(*this);
  switch (method) {
  case '+' :
   service.add(att, orbit, bf, bd, mux, muy, numtries, tolerance, stepsize);
   break;
  case '*' :
    service.multiply(att, orbit, bf, bd, mux, muy, numtries, tolerance, stepsize);
    break;
  default:
    service.multiply(att, orbit, bf, bd, mux, muy, numtries, tolerance, stepsize);
    break;
  }    
}

// Map

void Teapot::map(/*out*/ PacVTps& vtps, 
		 /*in*/ const PAC::BeamAttributes& att, 
		 /*in*/ int order)
{ 
  TeapotMapService service(*this);
  service.define(vtps, att, order);
}

void Teapot::trackMap(/*out*/ PacVTps& vtps, 
		      /*in*/ PAC::BeamAttributes& att, 
		      /*in*/ int i1, 
		      /*in*/ int i2)
{ 
  TeapotMapService service(*this);
  service.propagate(vtps, att, i1, i2);
}

void Teapot::transformOneTurnMap(/*out*/ PacVTps& output,
				 /*in*/ const PacVTps& oneTurnMap)
{
  TeapotMapService service(*this);
  service.transformOneTurnMap(output, oneTurnMap);
}

void Teapot::transformSectorMap(/*out*/ PacVTps& output,
				/*inout*/ PacVTps& oneTurnMap, 
				/*in*/ const PacVTps& sectorMap)
{
  TeapotMapService service(*this);
  service.transformSectorMap(output, oneTurnMap, sectorMap);
}

// Matrix

void Teapot::matrix(PacVTps& vtps, const PAC::BeamAttributes& att, const PAC::Position& delta)
{   
  TeapotMatrixService service(*this);
  service.define(vtps, att, delta, vtps.mltOrder());
}

void Teapot::decouple(const PAC::BeamAttributes& att, const PAC::Position& orbit,
		      const PacVector<int>& a11s,  const PacVector<int>& a12s,
		      const PacVector<int>& a13s,  const PacVector<int>& a14s,
		      const PacVector<int>& b1fs,  const PacVector<int>& b2fs,
		      double mux, double muy)
{
  TeapotMatrixService service(*this);
  service.decouple(att, orbit, a11s, a12s, a13s, a14s, b1fs, b2fs, mux, muy);
}		      

// Private methods

void Teapot::initialize()
{
  _nelem = 0;
  _telements = 0;

  _twissList = 0;

}

void Teapot::initialize(const PacLattice& l)
{
  initialize();

  _nelem = l.size();
  _telements = new TeapotElement[_nelem];
  if(!_telements){
    string msg = "Error : Teapot::initialize(const PacLattice& l) : allocation failure \n";
    PacAllocError(msg).raise();
  }

  for(int i=0; i < _nelem; i++) {
    _telements[i] = l[i];
  }
  
}

void Teapot::erase()
{
  if(_telements) delete [] _telements;
  eraseTwissList();
  initialize();

}



