// Library     : Teapot
// File        : Main/TeapotChromService.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Main/TeapotMapService.h"
#include "Main/TeapotClOrbitService.h"
#include "Main/TeapotMatrixService.h"
#include "Main/TeapotTwissService.h"
#include "Main/TeapotChromService.h"
#include "Main/Teapot.h"

// static int    TeapotChromService_Dim = 4;
// static double Delta   = 1.e-8;
static double DSTRENG = 4.e-4;
static int    ORDER   = 2;


TeapotChromService::TeapotChromService(Teapot& code)
  : code_(code), bfSize_(0), bdSize_(0), bfs_(0), bds_(0)
{
}

TeapotChromService::~TeapotChromService()
{
  closeMlt();
}

void TeapotChromService::define(PacChromData& chrom, const PAC::BeamAttributes& beam, const PAC::Position& orbit)
{
  // Find Map
  
  PacTMap map(TEAPOT_DIMENSION);
  map.refOrbit(orbit);

  TeapotMapService map_service(code_);
  map_service.define(map, beam, 2);

  // Find Twiss

  PacTwissData& twiss = chrom.twiss();

  TeapotTwissService twiss_service(code_);
  twiss_service.define(twiss, beam, orbit);

  // Find Chrom

  TeapotMatrix dr(4,4);
  PAC::Position d;
  d.set(twiss.d(0), twiss.dp(0), twiss.d(1), twiss.dp(1), 0.0, 1.0);

  findDR(dr, map, d);
  for(int id = 0, x = 0; id < 2; id++, x += 2){
    // std::cerr << x << 
    //  " dr[x][x]=" << dr[x][x] << 
    //  ", dr[x+1][x+1] = " << dr[x+1][x+1] << 
    //  ", sin = " << sin(twiss.mu(id)) << endl;
    chrom.dmu(id) = - (dr[x][x] + dr[x+1][x+1])/(2.*sin(twiss.mu(id)));
  }

}

void TeapotChromService::add(const PAC::BeamAttributes&, const PAC::Position&,
			     const PacVector<int>&, const PacVector<int>&, 
			     double, double,
			     int, double, double)
{
  
}

void TeapotChromService::multiply(const PAC::BeamAttributes& att, const PAC::Position& orbit0,
				  const PacVector<int>& bf, const PacVector<int>& bd, 
				  double mux, double muy,
				  int numtries, double tolerance, double stepsize)
{
  cout << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "fitting of chromaticity by varying cell sextupole strengths " << endl;
  cout << "desired chromaticities are " << mux << " " << muy << endl;

  TeapotClOrbitService orbit_service(code_);

  PAC::Position       orbit;
  PacChromData      target;

  openMlt(bf, bd);

  double qxtmp = 2*mux*PI, stepx = 0.0, qk1x = 0.0, qk1xref = 0.0, qsavex, delqx;
  double qytmp = 2*muy*PI, stepy = 0.0, qk1y = 0.0, qk1yref = 0.0, qsavey, delqy;

  double detr, r11, r12, r22, r21;

  orbit = orbit0;
  orbit_service.define(orbit, att);
  define(target, att, orbit);

  double dstreng = DSTRENG;
  if(stepsize) { dstreng = stepsize; }

  int nit;
  for(nit = 0; nit < numtries ; nit++) {

    if(nit) {

      stepx   = stepy = dstreng;

      qk1x   += stepx;
      qk1xref = multiplyBfs(qk1x/qk1xref);

      qsavex = target.dmu(0);
      qsavey = target.dmu(1);

      orbit = orbit0;
      orbit_service.define(orbit, att);
      define(target, att, orbit); 

      r11 = (target.dmu(0) - qsavex)/stepx;
      r21 = (target.dmu(1) - qsavey)/stepx;

      qk1x -= stepx;
      qk1y += stepy;   

      qk1xref = multiplyBfs(qk1x/qk1xref);
      qk1yref = multiplyBds(qk1y/qk1yref);

      orbit = orbit0;
      orbit_service.define(orbit, att);
      define(target, att, orbit); 

      r12 = (target.dmu(0) - qsavex)/stepy;
      r22 = (target.dmu(1) - qsavey)/stepy;    

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
    define(target, att, orbit); 

    delqx = qxtmp - target.dmu(0);
    delqy = qytmp - target.dmu(1);

    cout << "chromaticities are : " << target.dmu(0)/2/PI << " " << target.dmu(1)/2/PI  << endl;
    if(fabs(delqx) + fabs(delqy) < 2.*tolerance ) break;

  }

  if(nit == numtries) {
    cout << "fit did not converge in " << nit << " steps " << endl;
  }

  cout << nit+1 << " steps, final chromaticities are " << target.dmu(0)/2/PI << " " << target.dmu(1)/2/PI << endl;
  cout << "----------------------------------------------------------" << endl;
  closeMlt();

}


void TeapotChromService::findDR(TeapotMatrix& dr, const PacTMap& map, const PAC::Position& d)
{
  int size = map.size(), index = 0;
  double value;

  for(int k = 0; k < 4; k++){ 
    for(int i=0; i < 4; i++) { dr[k][i] = 0.0; }
    index = size;
    for(int l = 0; l < 4; l++) { 
      for(int m = l; m < size; m++) {
	value = map(k, ++index);
	dr[k][l] += value*d[m];
	if(m < 4) dr[k][m] += value*d[l];
      }
    }
  }

}

void TeapotChromService::openMlt(const PacVector<int>& ifs, const PacVector<int>& ids)
{
  closeMlt();

  PacElemAttributes* body;
  PacElemMultipole mlt(ORDER);

  bfSize_ = ifs.size() ;
  bfs_    = new PacElemMultipole*[bfSize_];

  int i;
  PacElemAttributes::iterator it;
  for(i = 0; i < ((int) ifs.size()); i++){ 

    body = &code_.element(ifs[i]).body();
    it   = body->find(PAC_MULTIPOLE);

    if(it != body->end()) { }
    else { 
      code_.element(ifs[i]).add(mlt); 
      body = &code_.element(ifs[i]).body();      
      it = body->find(PAC_MULTIPOLE);
    }

    bfs_[i] = (PacElemMultipole*) &(*it);

    if( bfs_[i]->order() < ORDER) { 
      cerr << "TeapotTwissService::openMlt(...) : element's mlt order < " 
	   << ORDER << "\n";
      string msg = " ";
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

    if( bds_[i]->order() < ORDER) { 
      cerr << "TeapotTwissService::openSets(...) : element's mlt order < " 
	   << ORDER << "\n";
      string msg = " ";
      PacDomainError(msg).raise();
    }
  }

}

double  TeapotChromService::multiplyBfs(double value)
{
  for(int i = 0; i < bfSize_; i++){
    if(fabs(bfs_[i]->kl(ORDER)) < TEAPOT_EPS)
      bfs_[i]->kl(ORDER) = TEAPOT_EPS;
    else
      bfs_[i]->kl(ORDER) *= value;
  }
  return bfs_[0]->kl(ORDER);
}

double TeapotChromService::getBf()
{
  return bfs_[0]->kl(ORDER);
}


double  TeapotChromService::multiplyBds(double value)
{
  for(int i = 0; i < bdSize_; i++){
    if(fabs(bds_[i]->kl(ORDER)) < TEAPOT_EPS)
      bds_[i]->kl(ORDER) = TEAPOT_EPS;
    else
      bds_[i]->kl(ORDER) *= value;
  }
  return bds_[0]->kl(ORDER);
}

double TeapotChromService::getBd()
{
  return bds_[0]->kl(ORDER);
}

void TeapotChromService::closeMlt()
{
  if(bfSize_) { delete [] bfs_; bfs_ = 0; bfSize_ = 0; }
  if(bdSize_) { delete [] bds_; bds_ = 0; bdSize_ = 0; }
}
