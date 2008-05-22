// Library     : Teapot
// File        : Main/Teapot.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_H
#define TEAPOT_H

#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"

#include "Optics/PacChromData.h"
#include "Integrator/TeapotElement.h"

class TeapotMapService;
class TeapotTrackService;
class TeapotMatrixService;
class TeapotClOrbitService;
class TeapotEigenService;
class TeapotFirstTurnService;


class Teapot : public PacSmf
{
public:

  // Friends

  friend class TeapotMapService;
  friend class TeapotTrackService;
  friend class TeapotMatrixService;
  friend class TeapotClOrbitService;
  friend class TeapotEigenService;
  friend class TeapotFirstTurnService;

  // Constructors & destructor
  
  Teapot() { initialize();}
  Teapot(const PacLattice& l) { initialize(l); }
 ~Teapot() { erase(); }

  // Commands

  void use(const PacLattice& l) { erase(); initialize(l); }

  void makethin() {  }

  // Tracking

  void track(PAC::Bunch& bunch, int turns = 1);
  void track(PAC::Bunch& bunch, int index1, int index2);

  // Survey

  void survey(PacSurveyData& s) { survey(s, 0, size()); }
  void survey(PacSurveyData& s, int index1, int index2);

  // Closed Orbit

  void clorbit(PAC::Position& orbit, const PAC::BeamAttributes& beam);
  void trackClorbit(PAC::Position& orbit, const PAC::BeamAttributes& beam, int i1, int i2);
  void steer(PAC::Position& orbit, const PAC::BeamAttributes& beam, 
	     const PacVector<int> ads, const PacVector<int> dets, int method, char plane);

  // First Turn
  
  void ftsteer(PAC::Position& orbit, const PAC::BeamAttributes& beam, 
	     const PacVector<int> hads, const PacVector<int> hdets,
             const PacVector<int> vads, const PacVector<int> vdets,
             double MaxAllowedDev, const PacTwissData& tw, const int method);     // for first turn

 // for first turn
  void twissList(PacTwissData& tw, const PAC::BeamAttributes& att, const PAC::Position& orbit);
  void eraseTwissList();


  // Twiss

  void twiss(PacTwissData& tw, const PAC::BeamAttributes& beam, const PAC::Position& orbit);
  void trackTwiss(PacTwissData& tw, const PacVTps& map);
  void tunethin(const PAC::BeamAttributes& beam, const PAC::Position& orbit,
		const PacVector<int>& bf, const PacVector<int>& bd, 
	        double mux, double muy, char method = '*', 
	        int numtries = 100, double tolerance = 1.e-6, double stepsize = 0.0);

  // Eigen parameters

  void eigenTwiss(/*out*/ PacTwissData& twiss, 
		  /*in*/ const PacVTps& map);
  void trackEigenTwiss(/*out*/ PacTwissData& twiss, 
		       /*in*/ const PacVTps& sector);

  // Chromaticity

  void chrom(PacChromData& ch, const PAC::BeamAttributes& att, const PAC::Position& orbit);
  void chromfit(const PAC::BeamAttributes& att, const PAC::Position& orbit,
		const PacVector<int>& bf, const PacVector<int>& bd, 
	        double mux, double muy, char method = '*', 
	        int numtries = 10, double tolerance = 1.e-4, double stepsize = 0.0);


  // Map
  
  void map(/*out*/ PacVTps& vtps, 
	   /*in*/ const PAC::BeamAttributes& beam, 
	   /*in*/ int order);
  void trackMap(/*out*/ PacVTps& vtps, 
		/*in*/ PAC::BeamAttributes& beam,
		/*in*/ int i1, 
		/*in*/ int i2);
  void transformOneTurnMap(/*out*/ PacVTps& output,
			   /*in*/ const PacVTps& oneTurnMap);
  void transformSectorMap(/*out*/ PacVTps& output,
			  /*inout*/ PacVTps& oneTurnMap,
			  /*in*/ const PacVTps& sectorMap);
  

  // Matrix

  void matrix(PacVTps& vtps, const PAC::BeamAttributes& beam, const PAC::Position& delta);
  void decouple(const PAC::BeamAttributes& beam, const PAC::Position& orbit, 
		const PacVector<int>& a11s,  const PacVector<int>& a12s,
		const PacVector<int>& a13s,  const PacVector<int>& a14s,
		const PacVector<int>& b1fs,  const PacVector<int>& b2fs,
		double mux, double muy);
  
  // Access

  int size() const { return _nelem; }
  TeapotElement& element(int index) { return _telements[index]; }
  const TeapotElement& element(int index) const { return _telements[index]; }
  
protected:


private:

  int _nelem;
  TeapotElement* _telements;

  PacTwissData* _twissList;     // For first turn

  void initialize();
  void initialize(const PacLattice& l);
  void erase();

};



#endif
