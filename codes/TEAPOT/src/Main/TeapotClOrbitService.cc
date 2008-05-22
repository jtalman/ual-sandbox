// Library     : Teapot
// File        : Main/TeapotClOrbitService.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Integrator/TeapotIntegrator.h"
// #include "Main/TeapotMapService.h"
#include "Main/TeapotMatrixService.h"
#include "Main/TeapotTwissService.h"
#include "Main/TeapotClOrbitService.h"
#include "Main/SlidingBumps.h"
#include "Main/Teapot.h"

static int     TeapotClOrbitService_Dim  = 4;
static int     TeapotClOrbitService_Nit  = 150;
static double  TeapotClOrbitService_Tiny = 1.e-10;
static double  Delta = 1.e-8;

TeapotClOrbitService::TeapotClOrbitService(Teapot& code)
 : code_(code), 
   aSize_(0), atwiss_(0) ,  adjusters_(0), 
   dSize_(0), dtwiss_(0) ,  detectors_(0)  
{
}

TeapotClOrbitService::~TeapotClOrbitService()
{
  closeSets();
}

void TeapotClOrbitService::define(PAC::Position& orbit, const PAC::BeamAttributes& att)
{

  // cerr << "closed orbit from tracking   delta = " << orbit.de() << "\n";

  // Make an one-turn second-order map

  PacTMap map(TEAPOT_DIMENSION);  
  map.refOrbit(orbit);

  //  TeapotMapService service(code_);
  //  int mltOrder = map.mltOrder();
  //  map.mltOrder(2);
  //  service.define(map, att);
  //  map.mltOrder(mltOrder);

  PAC::Position delta;
  delta.set(Delta, Delta, Delta, Delta, 0.0, Delta);

  TeapotMatrixService service(code_);
  service.define(map, att, delta, 1);

  // Make a periodic solution

  PAC::Position before, after;
  closedOrbit(before, map, 1);

  before += orbit;

  // Make several iterations

  int i, nit, flag = 0;
  double d;

  after = before;
  propagate(after, att, 0, code_.size());

  for(nit = 0; nit < TeapotClOrbitService_Nit; nit++) {

    before += after;
    before /= 2.;
    before[4] = orbit[4];
    before[5] = orbit[5];

    after = before;

    flag = propagate(after, att, 0, code_.size());  
    if(flag) break;

    d = 0;
    for(i=0; i < 4; i++) d += fabs(after[i] - before[i]);
    if(d < TeapotClOrbitService_Tiny) break;    
  }

  if(flag) {
    cerr << "\nparticle lost, closed orbit did not converge in " << nit << " turns\n";
    string msg = " ";
    PacDomainError(msg).raise();
  }

  if(nit == TeapotClOrbitService_Nit) {
    cerr << "\nclosed orbit did not converge in " <<  nit << " turns \n";
  }

  for(i=0; i < TeapotClOrbitService_Dim; i++) { orbit[i] = after[i]; }
 
}

void TeapotClOrbitService::define(PAC::Position& orbit, const PAC::BeamAttributes& beam,
				  PAC::Position* ps, const PacVector<int>& dets)
{
  // Find closed orbit

  define(orbit, beam);

  // Propagate it

  int i0 = 0;
  PAC::Position p(orbit);

  for(int id = 0; id < ((int) dets.size()); id++){ 
    propagate(p, beam, i0, dets[id]+1);
    ps[id] = p;
    i0 = dets[id] + 1;
  }

}

int TeapotClOrbitService::propagate(PAC::Position& p, const PAC::BeamAttributes& att, int index1, int index2)
{

  PAC::Position tmp(p);
  PAC::BeamAttributes beam = att;

  double e = att.getEnergy(), m = att.getMass();
  double v0byc = sqrt(e*e - m*m)/e;

  TeapotIntegrator integrator;  

  integrator.makeVelocity(p, tmp, v0byc);
  integrator.makeRV(att, p, tmp);

  int flag = 0;
  for(int j = index1; j < index2; j ++) {
    flag = integrator.propagate(code_._telements[j], p, tmp, beam, &v0byc);
    if(flag) break;
  } 
  return flag;
}

void TeapotClOrbitService::steer(PAC::Position& orbit0, const PAC::BeamAttributes& beam,
				 const PacVector<int>& ads, const PacVector<int>& dets, int BUMPS, char plane)
{ 
  if(BUMPS) steerSB(orbit0, beam, ads, dets, plane);
  else steerLS(orbit0, beam, ads, dets, plane);
}

const double PTOL = 1.0e-10;
const int ITMX1 = 50;

void TeapotClOrbitService::steerSB(PAC::Position& orbit0, const PAC::BeamAttributes& beam,
				 const PacVector<int>& ads, const PacVector<int>& dets, char plane)
{
  cout << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "Sliding Bumps orbit flattening by varying " << plane << "kick strengths" <<  endl;

   int i;
   int pln_index = (plane == 'v') ? 1 : 0;

   openSets(ads, dets);

   PAC::Position orbit;
   PacTwissData twiss;
   TeapotTwissService twiss_service(code_);

   // Make closed orbit
   orbit = orbit0;
   define(orbit, beam, detectors_, dets);

   cout << "initial closed orbit " <<  endl;
   for(i=0; i < 4; i++) cout << orbit[i] << " "; 
   cout << endl;
  
   float penold = 0.0;
   int iter = 0;

   while (1) {
     ++iter;
     float pentot = 1.0;
  
     // Make twiss
     twiss = twiss_service.define(atwiss_, ads,  beam, orbit);
     twiss_service.define(dtwiss_, dets, beam, orbit);
 
     SlidingBumps bumps(&ads, adjusters_, atwiss_, &dets, detectors_,  dtwiss_, pln_index, twiss.mu(pln_index));
     bumps.SetMonitorStatus(1); 
     bumps.CalculateCorrection(0, dSize_-1);
     bumps.ApplyCorrection();
     
     orbit = orbit0;
     define(orbit, beam, detectors_, dets);
     
     float rms = 0.;
  
     for(i=0; i<dSize_ ; i++) 
       rms += detectors_[i][2*pln_index]*detectors_[i][2*pln_index];
   
     rms = rms/dSize_ ;
     pentot += rms;

     float tol = 2.0 * fabs( (pentot - penold) / (pentot + penold) );
     penold = pentot;

     if (iter>ITMX1) {
       /*-- Orbit solution didn't converge after ITMX1 cycles */
       cout <<" ! ERROR ! Orbit non-convergent after "<<ITMX1 << " cycles"<< endl;
       break;
     } else if (tol>PTOL) {
       cout<<" Cycle "<<iter<<" with penalty "<< pentot<< "..."<< endl;
       /*-- return for one more cycle */
     } else {
       cout<<" Finished cycle "<<iter<<" with penalty "<< pentot<< "..."<< endl;    
       break; 
       /*-- within tolerance, so we're done */
     }  
  }
   
   cout << "final closed orbit " <<  endl;
   for(int j=0; j < 4; j++) cout << orbit[j] << " "; 
   cout << endl;
   
   closeSets();
   cout << "----------------------------------------------------------\n";
   cout << endl;
}  

void TeapotClOrbitService::steerLS(PAC::Position& orbit0, const PAC::BeamAttributes& beam,
				 const PacVector<int>& ads, const PacVector<int>& dets, char plane)
{ 
 
  cout << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "least squares orbit flattening by varying " << plane << "kick strengths" <<  endl;

  openSets(ads, dets);

  // Make closed orbit

  PAC::Position orbit; // , orbit0;

  orbit = orbit0;
  define(orbit, beam, detectors_, dets);

  cout << "initial closed orbit " <<  endl;
  for(int i=0; i < 4; i++) cout << orbit[i] << " "; 
  cout << endl;

  // Make twiss

  PacTwissData twiss;
  // orbit = orbit0;
  TeapotTwissService twiss_service(code_);

  twiss = twiss_service.define(atwiss_, ads,  beam, orbit);
  twiss_service.define(dtwiss_, dets, beam, orbit);

  knkmath(twiss, plane);

  orbit = orbit0;
  define(orbit, beam, detectors_, dets);

  cout << "final closed orbit " <<  endl;
  for(int j=0; j < 4; j++) cout << orbit[j] << " "; 
  cout << endl;
 
  closeSets();
  cout << "----------------------------------------------------------" << endl;
  cout << endl;
}  

void TeapotClOrbitService::closedOrbit(PAC::Position& p, const PacVTps& map, int order)
{
  int i, size = TeapotClOrbitService_Dim;

  TeapotMatrix matrix(size, size);
  TeapotVector delta(size), tmp(size);

  // 1./(matrix - 1)

  closedMatrix(matrix, map);

  // Linear approximation
  
  for(i = 0; i < size; i++) { tmp[i] = delta[i] = map(i, 0); }

  delta = matrix * delta;

  // cerr << "\nclosed orbit from first order matrix is \n";
  // for(i = 0; i < size; i++) { cerr << delta[i]; }
  // cerr << "\n";

  if(order < 2) {
    for(i = 0; i < size; i++) { p[i] = delta[i];} 
    return;
  }

  // Second-order approximation

  int index, dim = map.size();

  for(i = 0; i < size; i++) {
    index = dim; 
    for(int l = 0; l < dim; l++)
      for(int m = l;  m < dim; m++){
	  index++;
	  if(l < size && m < size) tmp[i] += map(i, index)*delta[l]*delta[m];
      }
  }

  delta = matrix * tmp; 

  // cerr << "\nclosed orbit including second order terms is \n";
  // for(i = 0; i < size; i++) { cerr << delta[i]; }
  // cerr << "\n"; 

  for(i = 0; i < size; i++) { p[i] = delta[i];}
}


void TeapotClOrbitService::closedMatrix(TeapotMatrix& matrix, const PacVTps& map)
{ 
  int dim = ( (int) (matrix.rows()/2.) ); 

  for(int i = 0; i < matrix.rows(); i++)
    for(int j = 0; j < matrix.columns(); j++) 
      matrix[i][j] = 0.0;

  double rdetr;
  for(int id = 0, x = 0, px = 1; id < dim; id ++, x += 2, px += 2) {

    rdetr = 1./((1. - map(x, x+1))*(1. - map(px, px + 1)) - map(x, px + 1)*map(px, x + 1));
    
    matrix[x ][x ]  = (1. - map(px, px + 1))*rdetr;
    matrix[x ][px]  = map(x, px + 1)*rdetr;
    matrix[px][x ]  = map(px, x + 1)*rdetr;
    matrix[px][px]  = (1. - map(x, x + 1))*rdetr;
  }
}



void TeapotClOrbitService::knkmath(const PacTwissData& twiss, char plane)
{
  int ip, ix;
  switch(plane){
  case 'h':
    ip = 0; ix = 0;
    break;
  case 'v':
    ip = 1; ix = 2;   
    break;
  default:
    ip = 0; ix = 0;   
  }

  double mu = twiss.mu(ip);
  double rdenom = 1./(2.*sin(mu/2.)), rmuda;

  TeapotMatrix ttad(aSize_, dSize_), rmab(aSize_, aSize_);

  int ia, ib, id;
  for(ia = 0; ia < aSize_; ia++){
    for(ib = 0; ib < aSize_; ib++) {
      rmab[ia][ib] = 0.0;
    }
  }

  for(id=0; id < dSize_; id++){
    for(ia=0; ia < aSize_; ia++){
      rmuda = atwiss_[ia].mu(ip) - dtwiss_[id].mu(ip);
      if(rmuda < 0.0) rmuda += mu;
      ttad[ia][id] = cos(mu/2. - rmuda)*sqrt(atwiss_[ia].beta(ip)*dtwiss_[id].beta(ip))*rdenom;
    }
  }

  for(ia = 0; ia < aSize_; ia++){
    for(ib = ia; ib < aSize_; ib++) {
      rmab[ia][ib] = 0.0;
      for(id = 0; id < dSize_; id++){
	 rmab[ia][ib] += ttad[ia][id]*ttad[ib][id];
      } 
    }
  }


  //     calculate a normalization factor

  double norm = 0.0;                 // , tnorm = 0.0;
  for(ia = 0; ia < aSize_; ia++){
    for(ib = 0; ib < aSize_; ib++) {
      norm += fabs(rmab[ib][ia]);
    }
  }
 
  double avg  = norm/aSize_/aSize_;  // , tavg = tnorm/aSize_/aSize_;
  double mult = 1.0/avg;             // , tmult = 1.0/tavg;


  //     normalize to unity

  for(ia = 0; ia < aSize_; ia++){
    for(ib = 0; ib < aSize_; ib++){
      rmab[ib][ia] *= mult;
    }
  }


  //     ---  call matrix inversion here to calculate rminvh
  //     ---matin2calculates the inverse matrix, which may not be necessary since
  //     the equations are solved only once.  This should be replaced by a
  //     linear equation solver eventually.

  //     first, fill in below diagonal

  for(ia = 0; ia < aSize_; ia++){
    for(ib = ia; ib < aSize_; ib++){
      rmab[ib][ia] = rmab[ia][ib];
    }
  }

   
  // CALL matinv(rmab, rminvh, nknkha(ics), maxknkha, ier)

  TeapotMatrix rminv = rmab.inverse();

  // test whether matrix is really the inverse

  /*  
  double ans;
  for(ia = 0; ia < aSize_; ia++){
    for(ib = 0; ib < aSize_; ib++){
      ans = 0.0;
      for(id = 0; id < aSize_; id++){
	ans += rmab[ia][id]*rminv[id][ib];
      }
      cerr << ia << " " << ib << " " << ans << "\n";
    }
  }
  */
  

  //     take out normalization factor

  for(ia = 0; ia < aSize_; ia++) {
    for(ib = 0; ib < aSize_; ib++){
      rminv[ib][ia] *= mult;
    }
  }

  //     --- evaluate right hand side of distortion equation, and the badness

  PacVector<double> rva(aSize_);

  for(ia = 0; ia < aSize_; ia++) {
    rva[ia] = 0.0;
    for(id = 0; id < dSize_; id++){
      rva[ia] += detectors_[id][ix]*ttad[ia][id];
    }
  }

  double badness = 0.0, rmsx;
  for(id = 0; id < dSize_; id++){ 
    // cout << " bpm " << id << " " << detectors_[id][ix] << endl;
    badness += (detectors_[id][ix]*detectors_[id][ix]);
  }
  rmsx = sqrt(badness/dSize_);

  // cout << "orbit badness = " <<  badness << endl;
  // cout << "rms orbit distortion before correction-measured at BPMs is " << rmsx << endl;

  //     ---  calculate adjustor strengths for orbit flattening

  PacVector<double> rqa(aSize_);

  for(ia = 0; ia < aSize_; ia++){
    rqa[ia] = 0.0;
    for(ib = 0; ib < aSize_; ib++){
      rqa[ia] += rminv[ia][ib]*rva[ib];
 
    }
  }

  addSets(rqa, plane);

}

void TeapotClOrbitService::openSets(const PacVector<int>& ads, const PacVector<int>& dets)
{
  closeSets();

  // Make adjusters

  PacElemAttributes* body;
  PacElemMultipole mlt(0);

  aSize_      = ads.size() ;
  adjusters_  = new PacElemMultipole*[aSize_];

  int i;
  PacElemAttributes::iterator it;
  for(i = 0; i < ((int) ads.size()); i++){ 

    body = &code_.element(ads[i]).body();
    it   = body->find(PAC_MULTIPOLE);

    if(it != body->end()) { }
    else { 
      code_.element(ads[i]).add(mlt); 
      body = &code_.element(ads[i]).body();      
      it = body->find(PAC_MULTIPOLE);
    }

    adjusters_[i] = (PacElemMultipole*) &(*it);

    if( adjusters_[i]->order() < 0) { 
      string msg = "TeapotClOrbitService::openSets(...) : element's mlt order < 0 \n";
      PacDomainError(msg).raise();
    }
  }  

  atwiss_  = new PacTwissData[aSize_];

  // Make detectors

  dSize_     = dets.size();
  dtwiss_    = new PacTwissData[dSize_];
  detectors_ = new PAC::Position[dSize_];

}

void TeapotClOrbitService::addSets(const PacVector<double>& rqa, char plane)
{
  int ia;
  switch(plane){
  case 'h':
    for(ia = 0; ia < aSize_; ia++) { adjusters_[ia]->kl(0)  += rqa[ia]; }
    break;
  case 'v':
    for(ia = 0; ia < aSize_; ia++) { adjusters_[ia]->ktl(0) -= rqa[ia]; }
    break;
  default:
    for(ia = 0; ia < aSize_; ia++) { adjusters_[ia]->kl(0)  += rqa[ia]; }
  }    

}

void TeapotClOrbitService::closeSets()
{
  if(aSize_) { 
    delete [] adjusters_; adjusters_ = 0;
    delete [] atwiss_;    atwiss_ = 0;
    aSize_ = 0;
  }

  if(dSize_) { 
    delete [] detectors_; detectors_ = 0;
    delete [] dtwiss_;    dtwiss_ = 0;
    dSize_ = 0;
  }
}

