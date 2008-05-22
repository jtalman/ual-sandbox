// Library     : Teapot
// File        : Main/TeapotMatrixService.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include <float.h>
#include "Main/TeapotTrackService.h"
#include "Main/TeapotMatrixService.h"
#include "Main/TeapotClOrbitService.h"
#include "Main/TeapotTwissService.h"
#include "Main/Teapot.h"

static int TeapotMatrixService_Size1 = 11;
static int TeapotMatrixService_Size2 = 21;

static double TeapotMatrixService_Delta  = 1.e-6;
// static int    TeapotMatrixService_Dim    = 6;

// Decouple stuff

// static int    TeapotMatrixService_Npars  = 6;

static int    TeapotMatrixService_Nresults  = 6;
static int    TeapotMatrixService_Nvars     = 6;
static int    TeapotMatrixService_Npars     = 4;

// static double TeapotMatrixService_Diff   = 0.0000001;
// static double TeapotMatrixService_Weight =  sqrt(200.);   

static double TeapotMatrixService_Weight    =  1.0;    
 
static int    TeapotMatrixService_Nit    = 25; // 1000;    
// static double Decouple_Tiny              = 1.e-12; // 1.e-6;
static double Decouple_Tolerance         = 1.e-6;

double TeapotMatrixService::pi_ = 3.14159265358979323846; 

TeapotMatrixService::TeapotMatrixService(Teapot& code)
  : code_(code), index_(0),  v0byc_(1.),
    linRays_(TeapotMatrixService_Size1),
    tmpRays_(TeapotMatrixService_Size1)
{ 
  adjusters_ = new PacElemMultipole**[TeapotMatrixService_Nvars];
  aSizes_    = new int[TeapotMatrixService_Nvars];
  aSigns_    = new double * [TeapotMatrixService_Nvars];
  for(int ia = 0; ia < TeapotMatrixService_Nvars; ia++){
    adjusters_[ia] = 0;
    aSizes_[ia] = 0;
    aSigns_[ia] = 0;
  }
}

TeapotMatrixService::~TeapotMatrixService()
{
  deleteDecoupleAdjusters();
  delete [] aSigns_;
  delete [] aSizes_;
  delete [] adjusters_;
}

void TeapotMatrixService::define(PacVTps& vtps, const PAC::BeamAttributes& att, 
				 const PAC::Position& delta, int order)
{

  int size;
  if(order < 2) { size = TeapotMatrixService_Size1; }
  else          { size = TeapotMatrixService_Size2; }

  // Generate rays for tracking to get transfer matrix

  PAC::Bunch rays(size);
  for(int ip = 0; ip < size; ip++)
    for(int i = 0; i < 6; i++)  
      rays[ip].getPosition()[i] =  vtps(i, 0);

  genRays(rays, att, delta, order);

  // Track rays

  TeapotTrackService service(code_);
  service.propagate(rays);

  // Get matrix

  getMatrix(vtps, rays, delta, order);
  
}

int TeapotMatrixService::start(const PAC::Position& orbit, const PAC::BeamAttributes& att, const PAC::Position& delta)
{

  delta_ = delta;

  // Generate rays for tracking to get transfer matrix

  int ip;
  for(ip = 0; ip < linRays_.size(); ip++){
      linRays_[ip].getPosition() =  orbit;
      linRays_[ip].setFlag(0); 
  }     

  genRays(linRays_, att, delta_, 1);

  // Initialize tmp data

  tmpRays_ = linRays_;
  double e = att.getEnergy(), m = att.getMass();
  v0byc_ = sqrt(e*e - m*m)/e;

  for(ip = 0; ip < linRays_.size(); ip++){
      integrator_.makeVelocity(linRays_[ip].getPosition(), tmpRays_[ip].getPosition(), v0byc_);
      integrator_.makeRV(linRays_.getBeamAttributes(), linRays_[ip].getPosition(), tmpRays_[ip].getPosition()); 
  }

  return index_ = 0;
}

int TeapotMatrixService::next(PacVTps& vtps, PAC::BeamAttributes& att)
{

  if(index_ >= code_.size()){
    string msg = "TeapotMatrixService.next(..) : current index >= number of elements \n";
    PacDomainError(msg).raise();
  }

  // Track rays

  for(int ip = 0; ip < linRays_.size(); ip++){
    if(linRays_[ip].getFlag()){
      string msg = "TeapotMatrixService.next(..) : lost particle \n";
      PacDomainError(msg).raise();
    }
    linRays_[ip].setFlag(
      integrator_.propagate(code_._telements[index_], linRays_[ip].getPosition(), 
			    tmpRays_[ip].getPosition(), att, &v0byc_));
  }

  // Get matrix

  getMatrix(vtps, linRays_, delta_, 1);

  return ++index_; 
}

int TeapotMatrixService::stop()
{
  return code_.size();
}

void TeapotMatrixService::decouple(const PAC::BeamAttributes& beam, const PAC::Position& orbit0, 
				   const PacVector<int>& a11s,  const  PacVector<int>& a12s,
				   const PacVector<int>& a13s,  const  PacVector<int>& a14s,
				   const PacVector<int>& bfs,   const  PacVector<int>& bds,
				   double mux, double muy)
{
  cout << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "decoupling  " << endl;

  // Detectors

  TeapotVector prvValues   = makeDecoupleValues(0.0, 0.0); 
  TeapotVector newValues   = makeDecoupleValues(0.0, 0.0);
  TeapotVector tmpValues   = makeDecoupleValues(0.0, 0.0);
  TeapotVector oldValues   = makeDecoupleValues(0.0, 0.0);
  TeapotVector goalValues  = makeDecoupleValues(mux, muy);
  TeapotVector step        = makeDecoupleStep();

  getDecoupleValues(newValues, beam, orbit0);

  cerr << "Required values : \n";
  printDecoupleValues(goalValues);

  cerr << "Initial values : \n";
  printDecoupleValues(newValues);

  oldValues = newValues;

  // Adjusters

  int i, j, npars = TeapotMatrixService_Npars;
  int nresults = TeapotMatrixService_Nresults;

  TeapotVector grad(npars);
  PacVector<double> deltas(npars);

  // makeDecoupleAdjusters(0, a11s);
  // makeDecoupleAdjusters(1, a12s);
  // makeDecoupleAdjusters(2, a13s);
  // makeDecoupleAdjusters(3, a14s);


  makeDecoupleFamilies(0, a11s, a13s, TMS_MINUS);
  makeDecoupleFamilies(2, a12s, a14s, TMS_MINUS);
  makeDecoupleAdjusters(4, bfs);
  makeDecoupleAdjusters(5, bds);

  // Main Loop

  // double level;
  // TeapotMatrix newA(npars, npars);
  double normAdj, normCorr, level, oldLevel, descentMin, descent = 1.0;
  double slope, initgrmg, currgrmg, det, hesDet, scale, sc2, num, den, ww;
  TeapotMatrix newA(nresults, npars);
  TeapotMatrix oldA(npars, npars);
  TeapotMatrix luA(npars, npars);
 
  int nit;   

  // scale = 1.0 / (2.0*M_PI*fabs(sin(2.0*M_PI*mux) + sin(2.0*M_PI*muy)));
  scale = 1.0 / (2.0*pi_*fabs(sin(2.0*pi_*mux) + sin(2.0*pi_*muy)));
  sc2 = scale * scale;

  TeapotVector diff        = oldValues - goalValues;
  det = diff[0] * diff[3] - diff[1] * diff[2];
  level = 0.5 * (sc2 * det + diff[4] * diff[4] + diff[5] * diff[5]);
  for (nit = 0; nit < TeapotMatrixService_Nit; nit++) {
     prvValues = newValues;
     for (i=0; i < npars; i++) {
	switch(i) {
	case 0:
	    addDecoupleAdjusters(0, 1.0e-06);
	    addDecoupleAdjusters(1, 1.0e-06);
	    break;
	case 1:
	    addDecoupleAdjusters(2, 1.0e-06);
	    addDecoupleAdjusters(3, 1.0e-06);
	    break;
	case 2:
	    addDecoupleAdjusters(4, 1.0e-06);
	    break;
	case 3:
	    addDecoupleAdjusters(5, 1.0e-06);
	    break;
	}
	getDecoupleValues(tmpValues, beam, orbit0);
	for (j=0; j < nresults; j++) {
	   newA[j][i] = (tmpValues[j] - oldValues[j]) * 1.0e+06;
	}
	grad[i] = sc2*(diff[3]*newA[0][i] + newA[3][i]*diff[0] -
		       diff[2]*newA[1][i] - newA[2][i]*diff[1]) +
	   diff[4]*newA[4][i] + diff[5]*newA[5][i];
	switch(i) {
	case 0:
	    addDecoupleAdjusters(0, -1.0e-06);
	    addDecoupleAdjusters(1, -1.0e-06);
	    break;
	case 1:
	    addDecoupleAdjusters(2, -1.0e-06);
	    addDecoupleAdjusters(3, -1.0e-06);
	    break;
	case 2:
	    addDecoupleAdjusters(4, -1.0e-06);
	    break;
	case 3:
	    addDecoupleAdjusters(5, -1.0e-06);
	    break;
	}
     }

     currgrmg = dot(grad, grad);
     if (nit == 0) initgrmg = currgrmg;

     if (sqrt(currgrmg / initgrmg) < Decouple_Tolerance) break;

     for (i=0; i < npars; i++) {
	for (j=0; j < npars; j++) {
	   oldA[i][j] = sc2*(newA[3][j]*newA[0][i] + newA[3][i]*newA[0][j] -
			     newA[2][j]*newA[1][i] - newA[2][i]*newA[1][j]) +
	      newA[4][j]*newA[4][i] + newA[5][j]*newA[5][i];
	}
     }

     // -- step should be to
     for (i=0; i < npars; i++) step[i] = -grad[i];

     luA  = oldA;
     hesDet = luA.luDecomp();
     if (hesDet == 0.0) break;
     luA.luBksb(step);

     // Reduce the amount of the correction if the new level
     // is larger than the old level.
     normAdj = normDecoupleAdjusters();
     normCorr = dot(step, step);
     slope = dot(grad, step);
     descentMin = DBL_EPSILON * sqrt(normAdj / normCorr);
     descent = 1.0;
     oldLevel = level;
     for ( ; ; ) {  // Infinite loop
	addDecoupleAdjusters(0,  descent * step[0]);
	addDecoupleAdjusters(1,  descent * step[0]);
	addDecoupleAdjusters(2,  descent * step[1]);
	addDecoupleAdjusters(3,  descent * step[1]);
	addDecoupleAdjusters(4,  descent * step[2]);
	addDecoupleAdjusters(5,  descent * step[3]);

	getDecoupleValues(newValues, beam, orbit0);

	diff  = newValues - goalValues;

	det = diff[0] * diff[3] - diff[1] * diff[2];
	level = 0.5 * (sc2 * det + diff[4] * diff[4] + diff[5] * diff[5]);

	if (level < oldLevel + 0.001 * descent * slope) break;

	if (descent <= descentMin) break;

	addDecoupleAdjusters(0, -descent * step[0]);
	addDecoupleAdjusters(1, -descent * step[0]);
	addDecoupleAdjusters(2, -descent * step[1]);
	addDecoupleAdjusters(3, -descent * step[1]);
	addDecoupleAdjusters(4, -descent * step[2]);
	addDecoupleAdjusters(5, -descent * step[3]);
	if (descent == 1.0) {
	   descent = -slope / (2.0 * (level - oldLevel - slope));
	} else {
	   descent *= 0.5;
	}
	cout << "***** Reducing the step size ( " << descent << " ) *****"
	     << endl;
     }
     for (num = den = 0.0, i = 0; i < npars; i++) {
	deltas[i] += descent * step[i];
	ww = newValues[i] - prvValues[i];
	num += ww * ww;
	den += newValues[i] * newValues[i];
     }
    
     cout <<  "Iteration " << nit << endl;
     printDecoupleValues(newValues);
     printDecoupleDeltas(deltas);

     if (descent <= descentMin) break;
     if (sqrt(num / den) < Decouple_Tolerance) break;

     oldValues = newValues;
  }


  // if(nit == TeapotMatrixService_Nit) {
  //  cerr << "\ndecoupling did not converge in " <<  nit << " turns \n";
  // }
  if (nit == TeapotMatrixService_Nit || descent <= descentMin) {
     cerr << "\ndecoupling did not converge in " <<  nit << " turns \n";
  }

  cout << "Final values: " << endl;
  printDecoupleValues(newValues);

  cout << "----------------------------------------------------------" << endl;
  cout << endl;

}

void TeapotMatrixService::makeDecoupleAdjusters(int ia, const PacVector<int>& indices)
{
  if(!indices.size()){
    deleteDecoupleAdjusters(ia);
    return;
  }

  PacElemAttributes* body;
  PacElemMultipole mlt(1);

  aSizes_[ia]    = indices.size() ;
  adjusters_[ia] = new PacElemMultipole*[aSizes_[ia]];
  aSigns_[ia] = new double [aSizes_[ia]];

  int i;
  PacElemAttributes::iterator it;
  for(i = 0; i < ( (int) indices.size()); i++){ 

    body = &code_.element(indices[i]).body();
    it   = body->find(PAC_MULTIPOLE);

    if(it != body->end()) { }
    else { 
      code_.element(indices[i]).add(mlt); 
      body = &code_.element(indices[i]).body();      
      it = body->find(PAC_MULTIPOLE);
    }

    adjusters_[ia][i] = (PacElemMultipole*) &(*it);
    aSigns_[ia][i] = 1.0;

    if( adjusters_[ia][i]->order() < 1) { 
      string msg = "TeapotTwissService::openMlt(...) : element's mlt order < 1 \n";
      PacDomainError(msg).raise();
    }
  }  


}
void TeapotMatrixService::makeDecoupleFamilies(int ia,
		const PacVector<int>& index1, const PacVector<int>& index2,
		TeapotMatrixService_Enum sign2)
{
  if(index1.size() + index2.size() == 0){
    deleteDecoupleAdjusters(ia);
    return;
  }

  PacElemAttributes* body;
  PacElemMultipole mlt(1);

  aSizes_[ia]        = index1.size();
  adjusters_[ia]     = new PacElemMultipole*[aSizes_[ia]];
  aSigns_[ia]        = new double [aSizes_[ia]];

  aSizes_[ia + 1]    = index2.size();
  adjusters_[ia + 1] = new PacElemMultipole*[aSizes_[ia + 1]];
  aSigns_[ia + 1]    = new double [aSizes_[ia + 1]];

  unsigned int i;
  PacElemAttributes::iterator it;
  for(i = 0; i < index1.size(); i++){

    body = &code_.element(index1[i]).body();
    it   = body->find(PAC_MULTIPOLE);

    if(it != body->end()) { }
    else { 
      code_.element(index1[i]).add(mlt); 
      body = &code_.element(index1[i]).body();      
      it = body->find(PAC_MULTIPOLE);
    }

    adjusters_[ia][i] = (PacElemMultipole*) &(*it);
    aSigns_[ia][i] = 1.0;

    if( adjusters_[ia][i]->order() < 1) { 
      string msg = "TeapotTwissService::openMlt(...) : element's mlt "
	  "order < 1 \n";
      PacDomainError(msg).raise();
    }
  }  

  for (i = 0; i < index2.size(); i++) {
    body = &code_.element(index2[i]).body();
    it   = body->find(PAC_MULTIPOLE);

    if (it != body->end()) {
    } else { 
      code_.element(index2[i]).add(mlt); 
      body = &code_.element(index2[i]).body();      
      it = body->find(PAC_MULTIPOLE);
    }

    adjusters_[ia + 1][i] = (PacElemMultipole*) &(*it);
    if (sign2 == TMS_PLUS) {
       aSigns_[ia + 1][i] = 1.0;
    } else {
       aSigns_[ia + 1][i] = -1.0;
    }

    if( adjusters_[ia + 1][i]->order() < 1) { 
      string msg = "TeapotTwissService::openMlt(...) : element's mlt "
	  "order < 1 \n";
      PacDomainError(msg).raise();
    }
  }
}

void TeapotMatrixService::printDecoupleDeltas(const PacVector<double>& deltas)
{
  cout << "Delta strengths : ";
  for(int i=0; i < ( (int) deltas.size()); i++){ cout << deltas[i] << " "; }
  cout << endl;
}

void TeapotMatrixService::addDecoupleAdjusters(int ia, double v)
{
  for(int i=0; i < aSizes_[ia]; i++){
    adjusters_[ia][i]->kl(1) += aSigns_[ia][i] * v;
  }
}
double TeapotMatrixService::normDecoupleAdjusters()
{
   double ans = 0.0, trm;
   for(int ia=0; ia < TeapotMatrixService_Nvars; ia++) {
      trm = 0.0;
      for(int i=0; i < aSizes_[ia]; i++) {
	 trm += adjusters_[ia][i]->kl(1) * adjusters_[ia][i]->kl(1);
      }
      ans += trm / aSizes_[ia];
   }
   return ans;
}

void TeapotMatrixService::deleteDecoupleAdjusters(int ia)
{
  if(adjusters_[ia]) delete [] adjusters_[ia];
  if(aSigns_[ia]) delete [] aSigns_[ia];
  adjusters_[ia] = 0;
  aSizes_[ia]    = 0;
  aSigns_[ia]    = 0;
}

void TeapotMatrixService::deleteDecoupleAdjusters()
{
  for(int ia = 0; ia < TeapotMatrixService_Nvars; ia++){
    deleteDecoupleAdjusters(ia);
  }
}

void TeapotMatrixService::getDecoupleValues (TeapotVector& values, const PAC::BeamAttributes& beam, 
					     const PAC::Position& orbit0)
{

  double weight = TeapotMatrixService_Weight;

  // Make closed orbit

  PAC::Position orbit = orbit0;

  TeapotClOrbitService orbit_service(code_);
  orbit_service.define(orbit, beam);

  // Find matrix

  PAC::Position delta;
  double Delta  = TeapotMatrixService_Delta;
  delta.set(Delta, Delta, Delta, Delta, 0.0, Delta);

  PacTMap map(TEAPOT_DIMENSION);
  map.refOrbit(orbit);

  define(map, beam, delta, 1);

  // Make twiss

  PacTwissData twiss;

  TeapotTwissService twiss_service(code_);
  twiss_service.define(twiss, beam, orbit);

  // e11
  values[0] = map(2, 0 + 1) + 
              map(1, 3 + 1);

  // e12
  values[1] = map(2, 1 + 1) - 
              map(0, 3 + 1);

  // e21
  values[2] = map(3, 0  + 1) - 
              map(1, 2  + 1);

  // e22
  values[3] = map(3, 1 + 1) + 
              map(0 , 2  + 1);

  // qx
  values[4] = weight*(twiss.mu(0)/2/PI);

  // qx
  values[5] = weight*(twiss.mu(1)/2/PI);
}

TeapotVector TeapotMatrixService::makeDecoupleStep()
{ 
  int npars = TeapotMatrixService_Npars;

  TeapotVector v(npars);
  for(int i= 0; i < npars; i++) v[i] = 1.0e-6; 
  return v;
}

TeapotVector TeapotMatrixService::makeDecoupleValues(double mux, double muy)
{ 
  double weight = TeapotMatrixService_Weight;

  // int npars = TeapotMatrixService_Npars;
  int nresults = TeapotMatrixService_Nresults;

  TeapotVector v(nresults);
  for(int i= 0; i < nresults; i++) v[i] = 0.0;
  v[4] = mux*weight;
  v[5] = muy*weight;
  return v;
}

void TeapotMatrixService::printDecoupleValues(const TeapotVector& values)
{

  double weight = TeapotMatrixService_Weight;
 
  cerr << " e11, e12, e21, e22 : " 
       << values[0] << " " << values[1] << " " << values[2] << " " << values[3] << "\n";
  cerr << " det(e)             : "
       << values[0]*values[3] - values[1]*values[2] << "\n";
  cerr << " qx, qy             : "
       << values[4]/weight << " " << values[5]/weight << "\n";
}

void TeapotMatrixService::genRays(PAC::Bunch& bunch, const PAC::BeamAttributes& att, 
				  const PAC::Position& delta, int order)
{
  bunch.getBeamAttributes().setMass(att.getMass());
  bunch.getBeamAttributes().setEnergy(att.getEnergy());
  bunch.getBeamAttributes().setCharge(att.getCharge());

  int index = 1, i;

  for(i = 0; i < 4; i++) {
    bunch[index++].getPosition()[i] += delta[i];
    bunch[index++].getPosition()[i] -= delta[i]; 
  } 

  //   --   delta ones  (9 - 10)

  bunch[index++].getPosition()[5] += delta[5];
  bunch[index++].getPosition()[5] -= delta[5]; 

  if(order < 2) return;

  //   --   double ones (11 - 12)

  bunch[index  ].getPosition()[0]  += delta[0];
  bunch[index++].getPosition()[1] += delta[1];
  bunch[index  ].getPosition()[2]  += delta[2];
  bunch[index++].getPosition()[3] += delta[3];


  //   --   double delta ones (13 - 16)

  for(i = 0; i < 4; i ++){
    bunch[index  ].getPosition()[i]  += delta[i];
    bunch[index++].getPosition()[5] += delta[5];
  }

  //    --  coupling ones (17 - 20)

  for(int ix = 0; ix <= 1; ix++) 
    for(int iy = 2; iy <= 3; iy++) { 
      bunch[index  ].getPosition()[ix]  += delta[ix];
      bunch[index++].getPosition()[iy]  += delta[iy];
    }

}

void TeapotMatrixService::getMatrix(PacVTps& map, const PAC::Bunch& bunch, 
				    const PAC::Position& delta, int order)
{
  int i1, i2, index = 0;

  for(i1 = 0; i1 < 6; i1++) { map(i1, 0) = bunch[index].getPosition()[i1]; }
  index++;

  for(i2 = 0; i2 < 4; i2++){
    for(i1 = 0; i1 < 4; i1++) {  
      map(i1, i2 + 1) = (bunch[index].getPosition()[i1] 
			 - bunch[index+1].getPosition()[i1])/(2.*delta[i2]) ;
    }
    index += 2;
  }

  //    ---  now get delta ones (9 and 10)

  i2 = 5;
  for(i1 = 0; i1 < 4; i1++) {
      map(i1, i2 + 1) = (bunch[index].getPosition()[i1] 
			 - bunch[index + 1].getPosition()[i1])/(2.*delta[i2]) ;
  }

  if(order < 2) return;

}

