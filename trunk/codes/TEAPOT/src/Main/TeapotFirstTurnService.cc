// Library     : Teapot
// File        : Main/TeapotFirstTurnService.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "Integrator/TeapotIntegrator.h"
// #include "Main/TeapotMapService.h"
#include "Main/TeapotMatrixService.h"
#include "Main/TeapotTwissService.h"
#include "Main/Teapot.h"
#include "Main/TeapotFirstTurnService.h"
#include "Main/SlidingBumps.h"

#include <fstream>



TeapotFirstTurnService::TeapotFirstTurnService(Teapot& code, double  MaxAllowedDev)
 : code_(code) 
{
  haSize_ = 0;
  hadjusters_ = 0;
  hatwiss_ = 0;

  hdSize_ = 0;
  hdetectors_ = 0;
  hdtwiss_ = 0;

  vaSize_ = 0;
  vadjusters_ = 0;
  vatwiss_ = 0;

  vdSize_ = 0;
  vdetectors_ = 0;
  vdtwiss_ = 0;

  MaxAllowedDeviation_ = MaxAllowedDev;
  hmaxdets_ = 0;
  vmaxdets_ = 0;

}
//---------------------------------------------------------------------------------------

TeapotFirstTurnService::~TeapotFirstTurnService()
{
  closeSets();
}

//---------------------------------------------------------------------------------------

 int TeapotFirstTurnService::define(PAC::Position& orbit, const PAC::BeamAttributes& beam,
	     PAC::Position* hps, const PacVector<int>& hdets, PAC::Position* vps, const PacVector<int>& vdets,
             int& max1, int& max2, char& max_pln)
{
  
  // Propagate it


 // Create and compose alldets vector that include both vertical and horiz. detectors

   PacVector<int> alldets;
   vector<char> type;
   vector <int>::const_iterator hi;
   vector <int>::const_iterator vi;
   
//   for(hi=(int *)hdets.begin(), vi=(int *)vdets.begin(); hi != hdets.end() && vi !=vdets.end(); ) {
for(hi = hdets.begin(), vi = vdets.begin(); hi != hdets.end() && vi !=vdets.end(); ) {
    if(*hi < *vi) {
          alldets.push_back(*hi);   
          type.push_back('h');
          hi++;
         }
    else if(*hi > *vi){
          alldets.push_back(*vi);
          type.push_back('v'); 
          vi++; 
	}
    else{ 
        alldets.push_back(*hi); 
        type.push_back('c');  
        hi++; vi++; 
    }
   }
 
 
      while(vi !=vdets.end()){
          alldets.push_back(*vi);
          type.push_back('v'); 
          vi++;}
     while(hi !=hdets.end()){
          alldets.push_back(*hi);
          type.push_back('h'); 
          hi++;} 
    
         
 


// Propagate a particle orbit monitor by orbit monitor and pick when the
// orbit deviation exceed allowable  

  int i0 = 0;
  PAC::Position p(orbit);
  PAC::Position p0;
 

   int ipln;

  cout << "AlldetsSize= " << alldets.size() << endl;

  int id, ih, iv;
  for(id=0, ih=0, iv=0; id < (int) alldets.size(); id++){
    
   p0 = p;
//    cout << " Point0; i= " <<id << endl;
   if(propagate(p, beam, i0, alldets[id]+1)){  //If particle out of aperture on this monitor
//     cout << " Point1 " << endl;
     if(type[id] == 'h') ipln= 0;
     else if (type[id] == 'v') ipln= 2;
     else ipln= 100;                         // for common plane detector

    if(ipln == 0) {
       max1 = ih-1;
       max2 = iv-1;
       max_pln = 'h'; // cout << " Point2 " << endl;
       }
    else if(ipln == 2){
       max1 = iv-1;
       max2 = ih-1;
       max_pln = 'v'; // cout << " Point3 " << endl;
       }
    else { max1 = ih-1;
           max2 = iv-1;  
           max_pln = 'c'; // cout << " Point4 " << endl;
	 }
     break;
   }
   
   else{   //If particle out of max allowed deviation  on this monitor

    if(type[id] == 'h') { hps[ih] = p; ih++; ipln= 0;}
    else if (type[id] == 'v') { vps[iv] = p; iv++; ipln= 2;}
    else  { hps[ih] = p; vps[iv] = p; ih++; iv++; ipln= 100;}


   if(type[id] != 'c' && p[ipln] > MaxAllowedDeviation_){
     if(ipln == 0 &&   ih > hmaxdets_) {
       max1 = ih-1;
       max2 = iv-1; // cout << " Point5 " << endl;
       max_pln = 'h';  
       break;
      }
    else if(ipln == 2 &&   iv > vmaxdets_){
       max1 = iv-1;
       max2 = ih-1;  // cout << " Point6 " << endl;
       max_pln = 'v';
       break;
     }
   }

  else if(type[id] == 'c'){
      if(p[0] > MaxAllowedDeviation_ &&   ih > hmaxdets_){
       max1 = ih-1;
       max2 = iv-1;
       max_pln = 'h'; // cout << " Point7 " << endl;
       break;  
     }
     else if(p[2] > MaxAllowedDeviation_ &&  iv > vmaxdets_){     
       max1 = iv-1;
       max2 = ih-1;
       max_pln = 'v'; // cout << " Point8 " << endl;
       break;   
     }
   }  
    i0 = alldets[id] + 1;
  }
  }
  
  
  cout << "hdetsSize= " << hdets.size() << " vdetsSize= " << vdets.size() << endl;
// limits to make final correction
  if(id == (int) alldets.size()) {
       max1 = hdets.size()-1;
       max2 = vdets.size()-1;
       max_pln = 'h';
     }

   return 0;

}


//---------------------------------------------------------------------------------------
/*
 int TeapotFirstTurnService::define(PacPosition& orbit, const PacBeamAttributes& beam,
	     PacPosition* hps, const PacVector<int>& hdets, PacPosition* vps, const PacVector<int>& vdets,
             int& max1, int& max2, char& max_pln)
{
  
  // Propagate it


 // Create and compose alldets vector that include both vertical and horiz. detectors

   PacVector<int> alldets;
   vector<char> type;
   vector <int>::iterator hi;
   vector <int>::iterator vi;
   
   for(hi=hdets.begin(), vi=vdets.begin(); hi != hdets.end() && vi !=vdets.end(); ) {

    if(*hi < *vi) {
          alldets.push_back(*hi);   
          type.push_back('h');
          hi++;
         }
    else if(*hi > *vi){
          alldets.push_back(*vi);
          type.push_back('v'); 
          vi++; 
	}
    else{ 
    cout << "First turn steering: Could not contain common hor/ver orbit monitor" << endl;
    return 1;
	}
 }
 
   if(type.back() == 'h') {
      while(vi !=vdets.end()){
          alldets.push_back(*vi);
          type.push_back('v'); 
          vi++;}}
   else {
      while(hi !=hdets.end()){
          alldets.push_back(*hi);
          type.push_back('h'); 
          hi++;} 
    }
         
 


// Propagate a particle orbit monitor by orbit monitor and pick when the
// orbit deviation exceed allowable  

  int i0 = 0;
  PacPosition p(orbit);

 

   int ipln;

  cout << "AlldetsSize= " << alldets.size() << endl;

  for(int id=0, ih=0, iv=0; id < alldets.size(); id++){
    
    propagate(p, beam, i0, alldets[id]+1);

    

    if(type[id] == 'h') { hps[ih] = p; ih++; ipln= PacPosition::X;}
    else { vps[iv] = p; iv++; ipln= PacPosition::Y;}


   if(p[ipln] > MaxAllowedDeviation_ ){
     if(ipln == PacPosition::X) {
       max1 = ih-1;
       max2 = iv-1;
       max_pln = 'h';
       break;
     }
    else {
       max1 = iv-1;
       max2 = ih-1;
       max_pln = 'v';
       break;
     }
   }

    i0 = alldets[id] + 1;
  }
  
  
  cout << "hdetsSize= " << hdets.size() << " vdetsSize= " << vdets.size() << endl;
// limits to make final correction
  if(id == alldets.size()) {
       max1 = hdets.size();
       max2 = vdets.size();
       max_pln = 'h';
     }

   return 0;

}
*/
//---------------------------------------------------------------------------------------


int TeapotFirstTurnService::propagate(PAC::Position& p, const PAC::BeamAttributes& att, int index1, int index2)
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


//--------------------------------------------------------------------------------------- 
//const int BUMPS = 0;  // 0-Grote method; 1-Sliding Bumps


void TeapotFirstTurnService::steer(PAC::Position& orbit, const PAC::BeamAttributes& beam,
	                        const PacVector<int>& hads, const PacVector<int>& hdets,
                                const PacVector<int>& vads, const PacVector<int>& vdets,
                                const PacTwissData& tw, const int BUMPS){


   if(BUMPS)  steerSB(orbit, beam, hads, hdets, vads, vdets, tw);
   else steerGrote(orbit, beam, hads, hdets, vads, vdets, tw);

 }

//--------------------------------------------------------------------------------------- 

void TeapotFirstTurnService::steerSB(PAC::Position& orbit, const PAC::BeamAttributes& beam,
	                        const PacVector<int>& hads, const PacVector<int>& hdets,
                                const PacVector<int>& vads, const PacVector<int>& vdets,
                                const PacTwissData& tw){

    openSets(hads, hdets, vads, vdets);

//  These two lines needed for Sliding Bumps method
  double mu_x = tw.mu(0), mu_y = tw.mu(1);

  std::cout << std::endl;
  std::cout << "----------------------------------------------------------" << std::endl;
  std::cout << "First Turn Correction by Sliding Bumps method started.    " << std::endl;


SlidingBumps hBumps(&hads, hadjusters_, hatwiss_, &hdets, hdetectors_,  hdtwiss_, 0, mu_x, FT);
SlidingBumps vBumps(&vads, vadjusters_, vatwiss_, &vdets, vdetectors_,  vdtwiss_, 1, mu_y, FT);


//Main correction loop

  int max1, max2;
  char max_plane;
  int hdtsmin=0;
  int vdtsmin=0;

  int it=1;
  std::cout << "Start of iteration loop " << std::endl ;

  int cycle_count=0;

  do { 
  std::cout << "Iteration: " << it++ << std::endl;

  if(define(orbit, beam, hdetectors_, hdets, vdetectors_, vdets, max1, max2, max_plane)) return;

  if(max_plane == 'h'){
      if(max1 == hmaxdets_) cycle_count++;
      else cycle_count=0;
      FindLimits(hdtsmin,hmaxdets_, max1);  
      FindLimits(vdtsmin,vmaxdets_, max2);
   }
  else {
      if(max1 == vmaxdets_) cycle_count++;
      else cycle_count=0;
      FindLimits(vdtsmin,vmaxdets_, max1);  
      FindLimits(hdtsmin,hmaxdets_, max2);
   }

  std::cout << "   FindLimits " << std::endl;

 
  std::cout << "-------------------------" << std::endl;
  std::cout << " Horizontal correction   " << std::endl;
  std::cout << "-------------------------" << std::endl;

  hBumps.SetMonitorStatus(1, hdtsmin, hmaxdets_);   
  hBumps.CalculateCorrection(hdtsmin, hmaxdets_);

  std::cout << "-------------------------" << std::endl;
  std::cout << " Vertical correction     " << std::endl; 
  std::cout << "-------------------------" << std::endl;

  vBumps.SetMonitorStatus(1, vdtsmin, vmaxdets_);   
  vBumps.CalculateCorrection(vdtsmin, vmaxdets_);

  hBumps.OpenLastBump(hmaxdets_);    
  vBumps.OpenLastBump(vmaxdets_);      

  std::cout << "  OpenLast " << std::endl;
  hBumps.ApplyCorrection();
  vBumps.ApplyCorrection();
  std::cout << "  Apply  " << std::endl;



  std::cout << " hdtsmax= " << hmaxdets_ << "  vdtsmax= " << vmaxdets_ << std::endl; 

  if(cycle_count == 4){
    std::cout << "The method in infinite loop. Correction stopped" << std::endl;  
    break;
  }
  cycle_count++;
  
  } while(hmaxdets_ != (int) hdets.size()  &&  vmaxdets_ != (int) vdets.size()-1);   

  
  closeSets();

  if(cycle_count == 4){ 
    std::cout <<  "First Turn Correction was unsuccesful" << std::endl;
  }
  else std::cout << "First Turn Correction done." << std::endl;

  
  std::cout << "----------------------------------------------------------" << std::endl;
  std::cout << std::endl;
}

//--------------------------------------------------------------------------------------- 


void TeapotFirstTurnService::steerGrote(PAC::Position& orbit, const PAC::BeamAttributes& beam,
	                        const PacVector<int>& hads, const PacVector<int>& hdets,
                                const PacVector<int>& vads, const PacVector<int>& vdets,
                                const PacTwissData& tw){

   openSets(hads, hdets, vads, vdets);


  std::cout << std::endl;
  std::cout << "----------------------------------------------------------" << std::endl;
  std::cout << "First Turn Correction by Grote's method started." << std::endl;
  
  
GroteSteer hSteer(&hads, hadjusters_, hatwiss_, &hdets, hdetectors_,  hdtwiss_, 0);
GroteSteer vSteer(&vads, vadjusters_, vatwiss_, &vdets, vdetectors_,  vdtwiss_, 1);


//Main correction loop

  int max1, max2;
  char max_plane;
  int hdtsmin=0;
  int vdtsmin=0;

  int it=1;
  std::cout << "Start of iteration loop " << std::endl ;

  do { 
  std::cout << "Iteration: " << it << std::endl;


  if(define(orbit, beam, hdetectors_, hdets, vdetectors_, vdets, max1, max2, max_plane)) return;

  //  PrintDetectors(outF);

  if(max_plane == 'h'){
      FindLimits(hdtsmin,hmaxdets_, max1);  
      FindLimits(vdtsmin,vmaxdets_, max2);
   }
  else {
      FindLimits(vdtsmin,vmaxdets_, max1);  
      FindLimits(hdtsmin,hmaxdets_, max2);
   }


 if(max_plane == 'h') {
 std::cout << "-------------------------" << std::endl;
 std::cout << " Horizontal correction   " << std::endl;
 std::cout << "-------------------------" << std::endl; 

  hSteer.CalculateCorrection(hmaxdets_);
  hSteer.ApplyCorrection();
}
else{
  std::cout << "-------------------------" << std::endl;
  std::cout << " Vertical correction     " << std::endl; 
  std::cout << "-------------------------" << std::endl;
  
    vSteer.CalculateCorrection(vmaxdets_);
    vSteer.ApplyCorrection();
  }


 


std::cout << " hdtsmax= " << hmaxdets_ << "  vdtsmax= " << vmaxdets_ << std::endl; 

} while(hmaxdets_ != (int) hdets.size()  &&  vmaxdets_ != (int) vdets.size()-1);    

  
 
  closeSets();
  std::cout << "First Turn Correction done." << std::endl;
  std::cout << "----------------------------------------------------------" << std::endl;
  std::cout << std::endl;


}
//---------------------------------------------------------------------------------------

void TeapotFirstTurnService::FindLimits(int& dtsmin, int& dtsmax, int dtsmax0) {

  

  if(dtsmax0 > dtsmax) dtsmin = dtsmax;
  else if (dtsmax0 < dtsmin) dtsmin = 0;       
 
  dtsmax = dtsmax0;


}
 
//---------------------------------------------------------------------------------------
          

void TeapotFirstTurnService::openSets(const PacVector<int>& hads, const PacVector<int>& hdets, 
                       const PacVector<int>& vads, const PacVector<int>& vdets)
{
  closeSets();

  // Make horizontal and vertical adjusters

  PacElemAttributes* body;
  PacElemMultipole mlt(0);

  haSize_      = hads.size() ;
  hadjusters_  = new PacElemMultipole*[haSize_];
  vaSize_      = vads.size() ;
  vadjusters_  = new PacElemMultipole*[vaSize_];

  PacElemAttributes::iterator it;
  for(unsigned int i = 0; i < hads.size(); i++){

    body = &code_.element(hads[i]).body();
    it   = body->find(PAC_MULTIPOLE);

    if(it != body->end()) { }
    else { 
      code_.element(hads[i]).add(mlt); 
      body = &code_.element(hads[i]).body();      
      it = body->find(PAC_MULTIPOLE);
    }

    hadjusters_[i] = (PacElemMultipole*) &(*it);

    if( hadjusters_[i]->order() < 0) { 
      std::string msg = "TeapotFirstTurnService::openSets(...) : element's mlt order < 0 \n";
      PacDomainError(msg).raise();
    }
  }  



 for(unsigned int i = 0; i < vads.size(); i++){

    body = &code_.element(vads[i]).body();
    it   = body->find(PAC_MULTIPOLE);

    if(it != body->end()) { }
    else { 
      code_.element(vads[i]).add(mlt); 
      body = &code_.element(vads[i]).body();      
      it = body->find(PAC_MULTIPOLE);
    }

    vadjusters_[i] = (PacElemMultipole*) &(*it);

    if( vadjusters_[i]->order() < 0) { 
      std::string msg = "TeapotFirstTurnService::openSets(...) : element's mlt order < 0 \n";
      PacDomainError(msg).raise();
    }
  }

 

  hatwiss_  = new PacTwissData[haSize_];
  vatwiss_  = new PacTwissData[vaSize_];
  int ind;

  for(int i = 0 ; i < haSize_; i++){
    ind=hads[i]; 
    hatwiss_[i] = code_._twissList[ind];
  }
  for(int i = 0 ; i < vaSize_; i++){
    ind=vads[i];  
    vatwiss_[i] = code_._twissList[ind];
  }

  // Make horizontal and vertical detectors


  hdSize_     = hdets.size();
  hdtwiss_    = new PacTwissData[hdSize_];
  for(int i = 0 ; i < hdSize_; i++){  ind=hdets[i]; hdtwiss_[i] = code_._twissList[ind]; }
        
  hdetectors_ = new PAC::Position[hdSize_];

  vdSize_     = vdets.size();
  vdtwiss_    = new PacTwissData[vdSize_];
  for(int i = 0 ; i < vdSize_; i++){ind=vdets[i];  vdtwiss_[i] = code_._twissList[ind];}
  vdetectors_ = new PAC::Position[vdSize_];

}
//---------------------------------------------------------------------------------------


void TeapotFirstTurnService::closeSets()
{
  if(haSize_) { 
    delete [] hadjusters_; hadjusters_ = 0;
    delete [] hatwiss_;    hatwiss_ = 0;
    haSize_ = 0;
  }

  if(hdSize_) { 
    delete [] hdetectors_; hdetectors_ = 0;
    delete [] hdtwiss_;    hdtwiss_ = 0;
    hdSize_ = 0;
  }
  if(vaSize_) { 
    delete [] vadjusters_; vadjusters_ = 0;
    delete [] vatwiss_;    vatwiss_ = 0;
    vaSize_ = 0;
  }

  if(vdSize_) { 
    delete [] vdetectors_; vdetectors_ = 0;
    delete [] vdtwiss_;    vdtwiss_ = 0;
    vdSize_ = 0;
  }

}


//---------------------------------------------------------------------------------------

void TeapotFirstTurnService::PrintDetectors(ostream& out, int plane){

 int i;

 if(plane) {

 for(i=0; i < vdSize_; i++) 
   out << "i= " << i << "  y = " << (vdetectors_[i])[2] << endl;
}

 else  {

 for(i=0; i < hdSize_; i++) 
   out << "detector i= " << i << "  x = " << (hdetectors_[i])[0] << endl;
}

}
