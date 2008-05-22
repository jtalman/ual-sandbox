// Library     : Teapot
// File        : Main/TeapotEigenService.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Main/TeapotClOrbitService.h"
#include "Main/TeapotMapService.h"
#include "Main/TeapotEigenService.h"
#include "Main/Teapot.h"


// Constructor
TeapotEigenService::TeapotEigenService(Teapot& code)
  : code_(code)
{
}

// Constructor
TeapotEigenService::~TeapotEigenService()
{
}

// Return the dimension of this service
inline int TeapotEigenService::dimension() const 
{
  return 2;
}


// Define initial twiss parameters from the one-turn map.
void TeapotEigenService::define(/*out*/ PacTwissData& twiss, 
				/*in*/ const PacVTps& map) const
{

  PAC::Position eta;
  // if(map.size() > PacPosition::DE) { closedEta(eta, map); } 

  TeapotMatrix matrix;

  int px, dim = min(map.size()/2, dimension());

  // double mu;
  double cos_mu, sin_mu, r_h_sum, r_h_dif, s;

  for(int id = 0, x = 0; id < dim; id ++, x += 2){ 

    px = x + 1;

    r_h_sum = (map(x, x + 1) + map(px, px + 1))/2.;
    r_h_dif = (map(x, x + 1) - map(px, px + 1))/2.;

    cos_mu = r_h_sum;

    s   = (map(x, px + 1) >= 0 ? 1. : -1.);
    sin_mu = s*sqrt(-map(x, px + 1)*map(px, x +1) - r_h_dif*r_h_dif);

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

// Define Twiss parameters
void TeapotEigenService::propagate(/*out*/ PacTwissData& in,
				   /*in*/ const PacVTps& map) const
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
      string msg = "Error : TeapotEigenService::propagate(...) beta == 0 \n";
      PacDomainError(msg).raise();
    }

    if(map_diff_1 == 0){    
      string msg = "Error : TeapotEigenService::propagate(...) map_diff_1 == 0 \n";
      PacDomainError(msg).raise();
    }

    inv_beta = 1./in.beta(id);

    twiss.beta(id) =  inv_beta*(map_diff_1*map_diff_1 + map(x, px + 1)*map( x, px + 1));
    twiss.alpha(id) = -inv_beta*(map_diff_1*map_diff_2 + map(x, px + 1)*map(px, px + 1));

    
    // twiss.mu(id)    = atan2(map(x, px + 1), map_diff_1);
    // if( twiss.mu(id) < 0) twiss.mu(id) += 2*PI;

    twiss.mu(id)    = asin(map(x, px + 1)/sqrt(twiss.beta(id)*in.beta(id)));
    twiss.mu(id)   +=  in.mu(id);

    for(int i = 0; i < eta.size(); i++){
       twiss.d(id)  += map(x,  i + 1) * eta[i];
       twiss.dp(id) += map(px, i + 1) * eta[i]; 
    }     

  }

  in = twiss;
}


// Find the closed eta (uncoupled case)  
void TeapotEigenService::closedEta(/*out*/ PAC::Position& p, 
				   /*in*/ const PacVTps& map) const
{
  int i, j, size = 2*dimension();

  TeapotMatrix matrix(size, size);
  TeapotVector delta(size);

  // 1./(matrix - 1)

  TeapotMatrix temp(matrix);

  for(i = 0; i < size; i++){
    for(j = 0; j < size; j++) {
      temp[i][j] = -map(i, j + 1);
    }
    temp[i][i] += 1.;
  }

  matrix = temp.inverse();

  // Linear approximation
  
  for(i = 0; i < size; i++) { delta[i] = map(i, 5 /*DE*/ + 1); }

  delta = matrix * delta;

  for(i = 0; i < size; i++) { p[i] = delta[i]; }

  p[4] = 0;
  p[5] = 1;

}
