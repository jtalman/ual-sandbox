// Library     : Teapot
// File        : Main/TeapotTwissService.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Main/TeapotClOrbitService.h"
#include "Main/TeapotMatrixService.h"
#include "Main/TeapotTwissService.h"
#include "Main/Teapot.h"

#include <stdlib.h>

static int    TeapotTwissService_Dim = 4;
static double Delta   = 1.e-8;
static double DSTRENG = 4.e-4;

TeapotTwissService::TeapotTwissService(Teapot& code)
  : code_(code),  bfSize_(0), bdSize_(0),  bfs_(0),  bds_(0)
{
}

TeapotTwissService::~TeapotTwissService()
{
  closeMlt();
}

void TeapotTwissService::define(PacTwissData& twiss, const PAC::BeamAttributes& att, const PAC::Position& orbit)
{

  // Get Map

  PAC::Position delta;
  delta.set(Delta, Delta, Delta, Delta, 0.0, Delta);

  PacTMap map(TEAPOT_DIMENSION);
  map.refOrbit(orbit);

  TeapotMatrixService matrix_service(code_);
  matrix_service.define(map, att, delta, 1);

  // Get Twiss
  
  PacTwissData tw;
  double mu[2] = {0, 0};

  define(twiss, map);

  // Track twiss

  int  id, qint[2] = {0, 0};

  int index = 0;
  PAC::BeamAttributes beam = att;
  matrix_service.start(orbit, att, delta);
  while( index !=  matrix_service.stop()){
    index = matrix_service.next(map, beam);
    tw = twiss;
    propagate(tw, map);

    for(id = 0; id < 2; id++) {
      if((tw.mu(id) - mu[id]) < -1.0e-6) { qint[id] += 1; }
      mu[id] = tw.mu(id);
    }
    
  }

  for(id = 0; id < 2; id++) {
      twiss.mu(id) = mu[id] + 2*qint[id]*PI;
  }

}

PacTwissData TeapotTwissService::define(PacTwissData* twiss, const PacVector<int> indices, 
				const PAC::BeamAttributes& att, const PAC::Position& orbit)
{

  // Get Map

  PAC::Position delta;
  delta.set(Delta, Delta, Delta, Delta, 0.0, Delta);

  PacTMap map(TEAPOT_DIMENSION);
  map.refOrbit(orbit);

  TeapotMatrixService matrix_service(code_);
  matrix_service.define(map, att, delta, 1);

  // Get Twiss
  
  PacTwissData tw0, tw;
  double mu[2] = {0, 0};

  define(tw0, map);

  // Track twiss

  int  id, qint[2] = {0, 0};

  int index = 0, vindex = 0, vsize = indices.size();

  PAC::BeamAttributes beam = att;
  matrix_service.start(orbit, att, delta);
  while( index !=  matrix_service.stop()){

    index = matrix_service.next(map, beam);

    tw = tw0;
    propagate(tw, map);

    for(id = 0; id < 2; id++) {
      if((tw.mu(id) - mu[id]) < -1.0e-6) { qint[id] += 1; }
      mu[id] = tw.mu(id);
    }

    if(vindex < vsize){
      if(index - 1 == indices[vindex]){

	tw.mu(0) = mu[0] + 2*qint[0]*PI;
	tw.mu(1) = mu[1] + 2*qint[1]*PI;
	twiss[vindex] = tw;      
	vindex++;

      }
    }    
  }

  for(id = 0; id < 2; id++) {
      tw0.mu(id) = mu[id] + 2*qint[id]*PI;
  }

  return tw0;
}

void TeapotTwissService::define(PacTwissData& twiss, const PacVTps& map)
{

  PAC::Position eta;
  if(map.size() > 5) { closedEta(eta, map); }

  int px, dim = min(map.size(), TeapotTwissService_Dim)/2;
  // double mu;
  double cos_mu, sin_mu, r_h_sum, r_h_dif, s;
  double xxx;

  for(int id = 0, x = 0; id < dim; id ++, x += 2){ 

    px = x + 1;

    r_h_sum = (map(x, x + 1) + map(px, px + 1))/2.;
    r_h_dif = (map(x, x + 1) - map(px, px + 1))/2.;

    cos_mu = r_h_sum;

    s   = (map(x, px + 1) >= 0 ? 1. : -1.);
    xxx = -map(x, px + 1)*map(px, x +1) - r_h_dif*r_h_dif;

    //======new block==================================
    if(xxx < 0.0) {
      cerr << " void TeapotTwissService::define \n";
      cerr << " wrong map , cos(mu) =" << cos_mu << " \n";
      cerr << " see map in the map_wrong.out file" << " \n";
      char file_name[] = "map_wrong.out";
      map->write(file_name);
      cerr << " STOP. TeapotTwissService. \n";
      exit(1);
    }
    //======new block==================================

    sin_mu = s*sqrt(xxx);

    if(cos_mu == 0) {
      cerr << id << " unstable, cos(mu) is 0 \n";
      string msg = " ";
      PacDomainError(msg).raise();
    }

    twiss.mu(id)    =  atan2(sin_mu, cos_mu);
    if(twiss.mu(id) < 0) twiss.mu(id) += 2*PI;

    twiss.beta(id)  =  map(x, px + 1)/sin_mu;
    twiss.alpha(id) =  r_h_dif/sin_mu;

    if(map.size() > 5) {

      twiss.d(id)   = eta[x];
      twiss.dp(id)  = eta[px]; 				   
    }

    twiss.mu(id)    = 0.0;

  }

}


void TeapotTwissService::propagate(PacTwissData& in, const PacVTps& map)
{

  int dim = min(map.size(), in.dimension());
  PacTwissData twiss(dim);

  double map_diff_1, map_diff_2, inv_beta;
  int px;

  PAC::Position eta;
  eta.set(in.d(0), in.dp(0), in.d(1), in.dp(1), 0, 1.);

  for(int id = 0, x = 0; id < dim; id ++, x += 2){ 

    px = x + 1;

    map_diff_1 = map( x, x + 1)*in.beta(id) - map( x, px + 1)*in.alpha(id);
    map_diff_2 = map(px, x + 1)*in.beta(id) - map(px, px + 1)*in.alpha(id);

    if(in.beta(id) == 0){    
      string msg = "Error : TeapotTwissService::propagate(...) beta == 0 \n";
      PacDomainError(msg).raise();
    }

    if(map_diff_1 == 0){    
      string msg = "Error : TeapotTwissService::propagate(...) map_diff_1 == 0 \n";
      PacDomainError(msg).raise();
    }

    inv_beta = 1./in.beta(id);

    twiss.beta(id) =  inv_beta*(map_diff_1*map_diff_1 + map(x, px + 1)*map( x, px + 1));
    twiss.alpha(id) = -inv_beta*(map_diff_1*map_diff_2 + map(x, px + 1)*map(px, px + 1));
    twiss.mu(id)    = atan2(map(x, px + 1), map_diff_1);
    if( twiss.mu(id) < 0) twiss.mu(id) += 2*PI;
    twiss.mu(id)   +=  in.mu(id);

    for(int i = 0; i < eta.size(); i++){
       twiss.d(id)  += map(x,  i + 1) * eta[i];
       twiss.dp(id) += map(px, i + 1) * eta[i]; 
    }     

  }

  in = twiss;

}

void TeapotTwissService::add(const PAC::BeamAttributes&, const PAC::Position&,
			     const PacVector<int>&, const PacVector<int>&, 
			     double, double,
			     int, double, double)
{
  
}


void TeapotTwissService::multiply(const PAC::BeamAttributes& att, const PAC::Position& orbit0,
				  const PacVector<int>& b1f, const PacVector<int>& b1d, 
				  double mux, double muy,
				  int numtries, double tolerance, double stepsize)
{
  cout << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "fitting of tunes by varying cell quad strengths " << endl;
  cout << "desired tunes are " << mux << " " << muy << endl;

  TeapotClOrbitService orbit_service(code_);

  PAC::Position       orbit;
  PacTwissData      twiss;

  openMlt(b1f, b1d);

  double qxtmp = 2*mux*PI, stepx = 0.0, qk1x = 0.0, qk1xref = 0.0, qsavex, delqx;
  double qytmp = 2*muy*PI, stepy = 0.0, qk1y = 0.0, qk1yref = 0.0, qsavey, delqy;

  double detr, r11, r12, r22, r21;

  orbit = orbit0;
  orbit_service.define(orbit, att);
  define(twiss, att, orbit);

  double dstreng = DSTRENG;
  if(stepsize) { dstreng = stepsize; }

  int nit;
  for(nit = 0; nit < numtries ; nit++) {

    if(nit) {

      stepx   = stepy = dstreng;

      qk1x   += stepx;
      qk1xref = multiplyBfs(qk1x/qk1xref);

      qsavex = twiss.mu(0);
      qsavey = twiss.mu(1);

      orbit = orbit0;
      orbit_service.define(orbit, att);
      define(twiss, att, orbit); 

      r11 = (twiss.mu(0) - qsavex)/stepx;
      r21 = (twiss.mu(1) - qsavey)/stepx;

      qk1x -= stepx;
      qk1y += stepy;   

      qk1xref = multiplyBfs(qk1x/qk1xref);
      qk1yref = multiplyBds(qk1y/qk1yref);

      orbit = orbit0;
      orbit_service.define(orbit, att);
      define(twiss, att, orbit); 

      r12 = (twiss.mu(0) - qsavex)/stepy;
      r22 = (twiss.mu(1) - qsavey)/stepy;    

      qk1y -= stepy;
      qk1yref = multiplyBds(qk1y/qk1yref);

      // ---  step should be to

      delqx = qxtmp - qsavex;
      delqy = qytmp - qsavey;

      detr = r11*r22 - r12*r21;

      stepx = (delqx*r22 - r12*delqy)/detr;
      stepy = (r11*delqy - r21*delqx)/detr;

      // ---  test printout of delta(k)  

      cout << "----------------------------------------------------------" << endl;
      cout << "old loop dkx, dky: " << nit << " " << stepx << " " << stepy << endl;
      cout << "r(i, j): " << r11 << " " << r12 << " " << r21 << " " << r22 << endl;

      // --- end of outer if loop

    }
    
    qk1x = getBf();
    qk1y = getBd();

    qk1xref = qk1x;
    qk1yref = qk1y;

    qk1x += stepx;
    qk1y += stepy;

    qk1xref = multiplyBfs(qk1x/qk1xref);
    qk1yref = multiplyBds(qk1y/qk1yref);

    orbit = orbit0;
    orbit_service.define(orbit, att);
    define(twiss, att, orbit); 

    delqx = qxtmp - twiss.mu(0);
    delqy = qytmp - twiss.mu(1);

    if(fabs(delqx) + fabs(delqy) < tolerance ) break;

  }

  if(nit == numtries) {
    cout << "fit did not converge in " << nit << " steps " << endl;
  }

  cout << nit+1 << " steps, final tunes are " << twiss.mu(0)/2/PI << " " << twiss.mu(1)/2/PI << endl;
  cout << "----------------------------------------------------------" << endl;
  closeMlt();

}

void TeapotTwissService::closedEta(PAC::Position& p, const PacVTps& map)
{
  int i, size = TeapotTwissService_Dim;

  TeapotMatrix matrix(size, size);
  TeapotVector delta(size);

  // 1./(matrix - 1)

  closedMatrix(matrix, map);

  // Linear approximation
  
  for(i = 0; i < size; i++) { delta[i] = map(i, 5 /*DE*/ + 1); }

  delta = matrix * delta;

  for(i = 0; i < size; i++) { p[i] = delta[i]; }

  p[4] = 0;
  p[5] = 1;

}

void TeapotTwissService::closedMatrix(TeapotMatrix& matrix, const PacVTps& map)
{
  //  int dim = matrix.rows()/2.; 

  TeapotMatrix temp(matrix);

  for(int i = 0; i < matrix.rows(); i++){
    for(int j = 0; j < matrix.rows(); j++) {
      temp[i][j] = -map(i, j + 1);
    }
    temp[i][i] += 1.;
  }

  matrix = temp.inverse();
  
  //  cerr << "Test inversion \n";
  //TeapotMatrix identity = matrix*temp;
  //for(int k = 0; k < matrix.rows(); k++) { cerr << identity[k][k] << " "; }
  //cerr << "\n";
  
}

void TeapotTwissService::openMlt(const PacVector<int>& ifs, const PacVector<int>& ids)
{
  closeMlt();

  PacElemAttributes* body;
  PacElemMultipole mlt(1);

  bfSize_ = ifs.size() ;
  bfs_    = new PacElemMultipole*[bfSize_];

  int i;
  PacElemAttributes::iterator it;
  for(i = 0; i <((int) ifs.size()); i++){ 

    body = &code_.element(ifs[i]).body();
    it   = body->find(PAC_MULTIPOLE);

    if(it != body->end()) { }
    else { 
      code_.element(ifs[i]).add(mlt); 
      body = &code_.element(ifs[i]).body();      
      it = body->find(PAC_MULTIPOLE);
    }

    bfs_[i] = (PacElemMultipole*) &(*it);

    if( bfs_[i]->order() < 1) { 
      string msg = "TeapotTwissService::openMlt(...) : element's mlt order < 1 \n";
      PacDomainError(msg).raise();
    }
  }  



  bdSize_ = ids.size() ;
  bds_    = new PacElemMultipole*[bdSize_];

  for(i = 0; i < ((int) ids.size()); i++){ 

    body = &code_.element(ids[i]).body();
    it   = body->find(PAC_MULTIPOLE);

    if(it != body->end()) { }
    else { 
      code_.element(ids[i]).add(mlt); 
      body = &code_.element(ids[i]).body();      
      it = body->find(PAC_MULTIPOLE);
    }

    bds_[i] = (PacElemMultipole*) &(*it);

    if( bds_[i]->order() < 1) { 
      string msg = "TeapotTwissService::openSets(...) : element's mlt order < 1 \n";
      PacDomainError(msg).raise();
    }
  }

}

double  TeapotTwissService::multiplyBfs(double b1)
{
  for(int i = 0; i < bfSize_; i++){
    if(fabs(bfs_[i]->kl(1)) < TEAPOT_EPS) 
      bfs_[i]->kl(1) = TEAPOT_EPS;
    else
      bfs_[i]->kl(1) *= b1;
  }
  return bfs_[0]->kl(1);
}

double TeapotTwissService::getBf()
{
  return bfs_[0]->kl(1);
}

double  TeapotTwissService::multiplyBds(double b1)
{
  for(int i = 0; i < bdSize_; i++){
    if(fabs(bds_[i]->kl(1)) < TEAPOT_EPS) 
      bds_[i]->kl(1) = TEAPOT_EPS;
    else
      bds_[i]->kl(1) *= b1;
  }
  return bds_[0]->kl(1);
}

double TeapotTwissService::getBd()
{
  return bds_[0]->kl(1);
}

void TeapotTwissService::closeMlt()
{
  if(bfSize_) { delete [] bfs_; bfs_ = 0; bfSize_ = 0; }
  if(bdSize_) { delete [] bds_; bds_ = 0; bdSize_ = 0; }
}
