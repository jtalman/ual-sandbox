// Library     : Teapot
// File        : Main/TeapotTrackService.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Integrator/TeapotIntegrator.h"
#include "Main/TeapotTrackService.h"
#include "Main/Teapot.h"

TeapotTrackService::TeapotTrackService(Teapot& code)
  : code_(code)
{
}

void TeapotTrackService::propagate(PAC::Bunch& bunch, int turns)
{
  // Calculate initial parameters

  PAC::Bunch tmp(bunch);
  PAC::BeamAttributes ba0(bunch.getBeamAttributes()), ba;

  ba = ba0;    //added by Shishlo

  // double total_l = getLength(ba0);

  int flag;
  TeapotIntegrator integrator; 

  for(int ip = 0; ip < bunch.size(); ip++){

    if(!bunch[ip].getFlag()){

      // Initialize beam attributes
      ba = ba0;
      double e = ba.getEnergy(), m = ba.getMass();
      double v0byc = sqrt(e*e - m*m)/e;
      // ba.revfreq(v0byc*BEAM_CLIGHT/total_l);

      // Initialize particle data
      integrator.makeVelocity(bunch[ip].getPosition(), tmp[ip].getPosition(), v0byc);
      integrator.makeRV(ba, bunch[ip].getPosition(), tmp[ip].getPosition());

      flag = 0;
      for(int i = 0; i < turns; i++)
	for(int j = 0; j < code_._nelem; j ++){
	  flag = integrator.propagate(code_._telements[j], bunch[ip].getPosition(), 
				      tmp[ip].getPosition(), ba, &v0byc); 
	  if(flag){
	    bunch[ip].setFlag(i+1);

            // PAC::Position& p = bunch[ip].getPosition();
            // printf("Lost: x= %11.4e y= %11.4e px= %11.4e py= %11.4e pc= %13.6e",
	    //	   p.getX(), p.getY(), p.getPX(), p.getPY(), p.getCT());
	    // string n = code_._telements[j].genElement().name();
	    // printf(" turn= %6d ie= %6d %10s\n", i, j, n.data());

	    j = code_._nelem;
	    i = turns;
	  }
	}
    } 
  }

  bunch.getBeamAttributes().setEnergy(ba.getEnergy());
  bunch.getBeamAttributes().setRevfreq(ba.getRevfreq());
}


void TeapotTrackService::propagate(PAC::Bunch& bunch, int index1, int index2)
{
  PAC::Bunch tmp(bunch);
  PAC::BeamAttributes ba0(bunch.getBeamAttributes()), ba;

  // double total_l = getLength(ba0);

  int flag;

  ba = ba0;    //added by Shishlo

  TeapotIntegrator integrator; 
  for(int ip = 0; ip < bunch.size(); ip++){
    if(!bunch[ip].getFlag()){

      // Initialize beam attributes
      ba = ba0;
      double e = ba.getEnergy(), m = ba.getMass();
      double v0byc = sqrt(e*e - m*m)/e;
      // ba.revfreq(v0byc*BEAM_CLIGHT/total_l);      

      integrator.makeVelocity(bunch[ip].getPosition(), tmp[ip].getPosition(), v0byc);
      integrator.makeRV(bunch.getBeamAttributes(), bunch[ip].getPosition(), tmp[ip].getPosition()); 

      flag = 0;
      for(int j = index1; j < index2; j++){
	flag = integrator.propagate(code_._telements[j], bunch[ip].getPosition(), tmp[ip].getPosition(), ba, &v0byc); 
        if(flag){
	  bunch[ip].setFlag(flag);
          j = index2;
	}
      }

    }
  }  

  bunch.getBeamAttributes().setEnergy(ba.getEnergy());
  bunch.getBeamAttributes().setRevfreq(ba.getRevfreq());
}

double TeapotTrackService::getLength(const PAC::BeamAttributes& ba0)
{

  PAC::BeamAttributes ba = ba0;

  // Define the original circumference

  PacSurveyData survey;
  code_.survey(survey);
  double l = survey.survey().suml();

  // Define the delay due to excess circumference

  PAC::Position orbit;
  code_.clorbit(orbit, ba);
  code_.trackClorbit(orbit, ba, 0, code_._nelem);
  double excess = - orbit[4];

  return l + excess;
}

