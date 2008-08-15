#include "SlidingBumps.h"
#include <math.h>
#include <sys/types.h>
#include <stdio.h> 
#include <stdlib.h>
#include <iostream>

typedef unsigned short u_short;

void CorrectorSet::Print(ostream& out, int plane) {

  int i;
  double str0; 

  for(i=0; i<corrNumber; i++){
    str0 = (plane) ?  strength0[i]->ktl(0) : strength0[i]->kl(0);

    out << i << "  " << (*latticeIndex)[i] << "  " << str0 <<  "  " << strength[i]  <<
      "   " << twiss[i].beta(plane) << "  " << twiss[i].mu(plane)<< endl;
  }

}


void MonitorSet::Print(ostream& out, int plane) {

  int i;
  // double str0; 
  
  for(i=0; i<monitNumber; i++){
    out << i << "  " << (*latticeIndex)[i] << "  " << 
      "   " << twiss[i].beta(plane) << "  " << twiss[i].mu(plane) << endl;
  }

}


//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------


ThreeBump::ThreeBump(){

  _plane = 0;
  _correctors = NULL;

  for(int i=0; i<3; i++){
    _corrIndex[i] = 0;
    _correctorResponce[i]=0;
  }

        _monitors = NULL;
        _monitIndex = NULL;
        _numberOfMonitors = 0;
        _monitorResponce = NULL;
}

//--------------------------------------------------------------------------------------

ThreeBump::ThreeBump(CorrectorSet* corrSet, int middleCorrInd, double mu_tot, int plane) { 

        _plane = plane;    
//initializes _correctorStrength and _strength     
        ConstructBump(corrSet, middleCorrInd, mu_tot, plane); 

               
        _monitors = NULL;
        _monitIndex = NULL;
        _numberOfMonitors = 0;
        _monitorResponce = NULL;
    
}

//--------------------------------------------------------------------------------------

ThreeBump::~ThreeBump(){
 
       delete [] _monitorResponce;
       delete [] _monitIndex;
     }

//--------------------------------------------------------------------------------------

ThreeBump& ThreeBump::operator= (const ThreeBump& inBump) {   

       int i;

       _plane = inBump._plane;
       _correctors = inBump._correctors;
       _monitors = inBump._monitors;

       for(i=0; i<3; i++){
         _corrIndex[i] = inBump._corrIndex[i];
         _correctorResponce[i] = inBump._correctorResponce[i];
       }

     _numberOfMonitors = inBump._numberOfMonitors;
      
        
     if(_monitIndex) delete [] _monitIndex;
     if(_monitorResponce) delete [] _monitorResponce;

     _monitIndex = new int[_numberOfMonitors];
     _monitorResponce = new float[_numberOfMonitors];

     for(i=0; i<_numberOfMonitors; i++){
      _monitIndex[i] = inBump._monitIndex[i];
      _monitorResponce[i] = inBump._monitorResponce[i];
     } 

     return *this;

}

//--------------------------------------------------------------------------------------

int ThreeBump::ConstructBump(CorrectorSet* corr, int middleCorrInd, double mu_tot, int plane) {


// Good for ring not for channel !!   
// Construct bump on the basis of middle corrector.
// Checking for good phase advances between correctors: MIN_SIN parameter

  //  int i;  

      _plane = plane;
      _correctors  = corr;

        int corrNumber = _correctors->corrNumber; 
        if( middleCorrInd >= corrNumber  || middleCorrInd < 0)
           cout << "Error at 3-bump creation. Corrector index is unreal" << endl;
        
        
    int i1 = middleCorrInd;
    double bhcl, bhcr, bhc  = _correctors->twiss[i1].beta(_plane);
    double fihcl, fihcr, fihc  = _correctors->twiss[i1].mu(_plane);
    

    int i0 = i1;
    int i2 = i1;

    const float MIN_SIN = 0.5;
    u_short count=0;

    do{ 
      count++; 
      if((--i0) < 0)  i0 = corrNumber - 1;
      bhcl = _correctors->twiss[i0].beta(_plane);
      fihcl = _correctors->twiss[i0].mu(_plane);
      if(i0>i1) fihcl -= mu_tot;               //  if the bump overlap the ring initial point
    }while(fabs(sin(fihc - fihcl)) < MIN_SIN && count < 3);

   count=0;
   do{
      count++; 
      if((++i2) == corrNumber)  i2 = 0;
      bhcr = _correctors->twiss[i2].beta(_plane);
      fihcr = _correctors->twiss[i2].mu(_plane);
      if(i1>i2) fihcr += mu_tot;               //  if the bump overlap the ring initial point
   }while(fabs(sin(fihc - fihcr)) < MIN_SIN && count < 3);

  if(fabs(sin(fihc - fihcl)) < MIN_SIN || fabs(sin(fihcr - fihc)) < MIN_SIN){
     cout << "Cannot construct bump for corrector " << (*_correctors->latticeIndex)[i1] << endl;
     return 1;   
   }
  
    _corrIndex[0] = i0;
    _corrIndex[1] = i1;
    _corrIndex[2] = i2;
    _correctorResponce[0] = (1.0 / sqrt(bhcl * bhc)) * (1.0 / sin(fihc - fihcl));
    _correctorResponce[1] = (1.0 / bhc) *( sin(fihcl-fihcr) / ( sin(fihcr-fihc) * sin(fihc-fihcl) ) );
    _correctorResponce[2] = (1.0 / sqrt(bhcr * bhc)) * (1.0 / sin(fihcr - fihc));  

    return 0;     
  
}

//--------------------------------------------------------------------------------------

const float RESP_LIM = 0.5;

void ThreeBump::FindMonitors(MonitorSet* monitors, double mu_tot) {   



   int i;

  _monitors = monitors;

//  Find all monitors inside the bump  

  int* tmpMonitor = new int[10];
  _numberOfMonitors = 0;
  int i0 = _corrIndex[0];
  int i1 = _corrIndex[1];
  int i2 = _corrIndex[2]; 
 
  if(i0>i1 || i1>i2 ) {       // If the bump is first or last  bump in the ring  
    for(i = 0; i < _monitors->monitNumber ; i++) {
      if((*_monitors->latticeIndex)[i] > (*_correctors->latticeIndex)[i0] ||  
	 (*_monitors->latticeIndex)[i] < (*_correctors->latticeIndex)[i2]) {
	tmpMonitor[_numberOfMonitors] = i ;
	_numberOfMonitors++;
      } 
      if(_numberOfMonitors == 10) {
	cout << "Too many monitors inside 3-bump at " << _corrIndex[1] << endl;
	break;
      }
    }
  }
  else {
    for(i = 0; i < _monitors->monitNumber ; i++) {
      
      if((*_monitors->latticeIndex)[i] > (*_correctors->latticeIndex)[i0] && 
	 (*_monitors->latticeIndex)[i] < (*_correctors->latticeIndex)[i2]) {
	tmpMonitor[_numberOfMonitors] = i ;
	_numberOfMonitors++;
      } 

      if(_numberOfMonitors == 10) {
	cout << "Too many monitors inside 3-bump at " << _corrIndex[1] << endl;
	break;
      }
    }
  }

// Calculate orbit responce: orbit on monitors when orbit at central corrector is 1mm
  
    float* responceTmp = new float[_numberOfMonitors];
    u_short* flag = new u_short[_numberOfMonitors];
    
    // double bhcl = _correctors->twiss[i0].beta(_plane);
    double bhc  = _correctors->twiss[i1].beta(_plane);
    //    double bhcr = _correctors->twiss[i2].beta(_plane);
    double fihcl = _correctors->twiss[i0].mu(_plane);
    double fihc  = _correctors->twiss[i1].mu(_plane);
    double fihcr = _correctors->twiss[i2].mu(_plane);

    if(i0>i1) fihcl -= mu_tot;       //for bumps
    else if(i1>i2) fihcr += mu_tot;  // overlaping the closing ring point

    double bhm, fihm;
  
     int j, count=0;;

   for(i=0; i< _numberOfMonitors; i++) {

       j = tmpMonitor[i];
       fihm = _monitors->twiss[j].mu(_plane);
       bhm = _monitors->twiss[j].beta(_plane);

       if(i0>i1 && fihm>fihcr ) fihm -= mu_tot;     
       else if(i1>i2 && fihm<fihcl) fihm += mu_tot; 

     if (fihcl<=fihm && fihm<=fihc) {
	/*-- the monitor is between left and center correctors */
	responceTmp[i] = sqrt(bhm / bhc) * sin(fihm - fihcl) / sin(fihc - fihcl);
      } else if (fihc<fihm && fihm<=fihcr) {
	/*--  the monitor is between center and right correctors, */
	responceTmp[i] = sqrt(bhm / bhc) * sin(fihcr - fihm) / sin(fihcr - fihc);
      } else {
	/*-- the monitor is outside the bump, an ERROR */
	cout << " !! ERROR !! monitor " << (*_monitors->latticeIndex)[i] << 
             " outside bump " << _corrIndex[1] << endl;
        responceTmp[i] = 0.;
      }
     
 // Exclude monitors with bad phase relation

        if(fabs(responceTmp[i]) < RESP_LIM) flag[i] = 0; 
        else { flag[i]=1; count++;}
     }

     if(_monitIndex) delete [] _monitIndex;
    _monitIndex = new int[count];

     if(_monitorResponce) delete [] _monitorResponce; 
      _monitorResponce = new float[count];

     

     j=0;
     for(i = 0; i < _numberOfMonitors; i++){

       if(flag[i]){
        _monitIndex[j] = tmpMonitor[i];
        _monitorResponce[j] = responceTmp[i];
        j++;
        }
       }

     _numberOfMonitors = count;

     delete [] tmpMonitor;
     delete [] responceTmp;
     delete [] flag;


}  
// --------------------------------------------------------------------------------------------

// Boolean ThreeBump::IfCrossed0(void) {     
bool ThreeBump::IfCrossed0(void) {  

  // int i1 = _corrIndex[0];
  // int i2 = _corrIndex[1];
  // int i3 = _corrIndex[2]; 

   if(_corrIndex[0]>_corrIndex[1] || _corrIndex[1]>_corrIndex[2] ) return true;
   else return false;
 }


// --------------------------------------------------------------------------------------------

 void ThreeBump::Print(ostream& out){

   int i1 = _corrIndex[0];
   int i2 = _corrIndex[1];
   int i3 = _corrIndex[2];

   out << "Corr.1: ";
    PrintCorrector(out, i1);
   out << "Corr.2: ";
    PrintCorrector(out,i2);
   out << "Corr.3: ";
    PrintCorrector(out,i3);


   for(int i=0; i<_numberOfMonitors; i++) {
   
   i1 = _monitIndex[i];
   out << "Monit." << i+1 << ": ";
    PrintMonitor(out, i1);
  }
    
}
// --------------------------------------------------------------------------------------------

void ThreeBump::PrintCorrector(ostream& out, int i){
 
   if(i >= _correctors->corrNumber) {
   cout << "Can not print corrector data. Out of range." << endl;
   return;
 }

  out << (*_correctors->latticeIndex)[i] << "  " << _correctors->strength[i]  <<
  "   " << _correctors->twiss[i].beta(_plane) << "  " << _correctors->twiss[i].mu(_plane)<< endl;

}

// --------------------------------------------------------------------------------------------

void ThreeBump::PrintMonitor(ostream& out, int i){
 
   if(i >= _monitors->monitNumber) {
   cout << "Can not print monitor data. Out of range." << endl;
   return;
 }

  out << (*_monitors->latticeIndex)[i] << "  " << (_monitors->position[i])[2*_plane]  <<
  "   " << _monitors->twiss[i].beta(_plane) << "  " << _monitors->twiss[i].mu(_plane)<< endl;

}
// --------------------------------------------------------------------------------------------
const double STRC0 = 0.1;
const int ITMX0 = 1000;
const double PFIN =  0.9;
const double FTOL = 1.0e-10;
const int ITW  =  -1;


double ThreeBump::Optimization() {
    
	double strength = STRC0;
	int itmx = ITMX0;

        int iprnt = 0; // ???
        int j,i;



        if(_numberOfMonitors != 1){
	onedim(&strength,&itmx,ITW,iprnt,PFIN,FTOL);
	/*-- optimize the amplitude of the bump */

	if (itmx < 0)
          cout<< " ! ERROR ! bump "<< _corrIndex[1]<< " unconverged after " <<
          itmx << " iters" << endl;
	/*-- single bump solution didn't converge ! */
      }
      else {                          
        j = _monitIndex[0];
        strength = -(_monitors->position[j])[2*_plane]/_monitorResponce[0];
      }


	double penal = PenaltyFunction(strength);
	//	int k; 

	for (i=0; i < _numberOfMonitors; ++i) {
	  /*-- accumulate the net closed orbit monitor values */
           j = _monitIndex[i];
          if(_monitors->status[j])     
         (_monitors->position[j])[2*_plane] += strength * _monitorResponce[i];
	}

	for (i=0; i < 3; ++i) {
	  /*-- accumulate the net corrector strengths */
          j = _corrIndex[i];
	  _correctors->strength[j] += strength * _correctorResponce[i];
          

//          k = (*_correctors->latticeIndex)[j];
//          if(k==11977 || k == 12033 || k==12089 || k==12145 || k==12201) {
//           cout << "k= " << k << "  corrstr= " << _correctors->strength[j] << 
//                   "  bumpstr= " << strength << "        corrResp= " << _correctorResponce[i] <<  endl;
//	 }
	}

        return penal;
 }



/*====================================================================== */

double ThreeBump::PenaltyFunction(double strbmp)
{
  /*---------------------------------------------------------------------------
  //   This is a penalty function called by ONEDIM to estimate
  // how well a particular amplitude bump fits the desired closed orbit,
  // while minding that the correctors do not go out of range.
  //-------------------------------------------------------------------------*/
  int    i, j;
  double penal = 1.0;
  double dco;
 
  /*--  loop over the MONITORS in the bump */
  for (i=0; i < _numberOfMonitors; ++i) {
    j = _monitIndex[i];
    if(_monitors->status[j]) {                
      dco = (_monitors->position[j])[2*_plane] + strbmp * _monitorResponce[i];
      penal += dco*dco;
    }
    /*-- ... and accumulate a penalty proportional to the square of the */
    /*-- remaining error, weighted by _weight, typically = 1.            */
  }
  
  return penal;
}


void ThreeBump::onedim(double *psol, int *itmx, int itw, int iprnt,
	    double pfin, double ftol)
{
  int    iter, ihi, ilo;
  double pr, rtol, ypr, prr, yprr;
  double p[3], y[3];

  //----------------------------------------------------------------------------
  // ONEDIM is the one dimensional version of HYDRA.
  // See "Numerical Recipes", Press et al., p292.
  //
  // PENALTYFUNCTION is the name of the function to be minimised
  // PSOL is the initial guess and final solution
  // ITMX is the maximum number of iterations allowed.  Negative on
  //      return if maximum number occurred without convergence.
  // ITW is the period with which ONEDIM reports back to unit IODATA.IPRNT
  // PFIN sets the size of the initial simplex
  // FTOL is the goal tolerance of PENALTYFUNCTION
  //
  // P(2) are the two current guesses
  // Y are the function values corresponding to the P values
  //
  // ALPHA, BETA and GAMMA are transformation scale factors
  //--------------------------------------------------------------------------
  const double ALPHA  = 1.0;
  const double BETA   = 0.5;
  const double GAMMA  = 2.0;
  const double DELTA  = 0.5;

  p[1] = (*psol);
  p[2] = pfin * (*psol);
  y[1] = PenaltyFunction(p[1]);
  y[2] = PenaltyFunction(p[2]);

  if (itw > 0)
    printf("  iter   rtol      y(ilo)      y(ihi)      p(ilo)      p(ihi)\n");

  iter = -1;

  /*-- beginning of iteration loop */
  while (1) {
    ++iter;

    /*-- determine which point is highest, and lowest */
    if (y[1] > y[2]) {
      ihi = 1;
      ilo = 2;
    } else {
      ihi = 2;
      ilo = 1;
    }

    rtol = 2.0 * fabs(y[ihi] - y[ilo]) / ( fabs(y[ihi]) + fabs(y[ilo]) );

    /*-- write output if appropriate */
    if (itw > 0) {
      if(iter%itw==0 || rtol<ftol || iter==(*itmx)) {
	printf("%5d %9.2f %14.5f %14.5f %14.5f %14.5f\n",
	       iter, rtol, y[ilo], y[ihi], p[ilo], p[ihi]);
      }
    }

    /*-- check for an acceptable solution */
    if (rtol<ftol || iter==(*itmx)) {
      (*psol) = p[ilo];
      if (iter == (*itmx)) (*itmx) *= -1;
      return;
    }

    /*-- "reflect" the highest point by ALPHA through the center */
    pr = p[ilo] + ALPHA * (p[ilo] - p[ihi]);
    ypr = PenaltyFunction(pr);

    /*-- if an ALPHA step was good, try another GAMMA step ... */
    if (ypr < y[ilo]) {
      prr = p[ilo] + GAMMA * (pr - p[ilo]);
      yprr = PenaltyFunction(prr);
      if (yprr < y[ilo]) {
	p[ihi] = prr;
	y[ihi] = yprr;
      } else {
	p[ihi] = pr;
	y[ihi] = ypr;
      }

    } else if (ypr > ( y[ilo]+DELTA*(y[ihi]-y[ilo]) ) ) {
      /*-- ... but if an ALPHA step was no good, contract about low point ... */
      prr = p[ihi] + BETA * (p[ilo] - p[ihi]);
      yprr = PenaltyFunction(prr);
      p[ihi] = prr;
      y[ihi] = yprr;

      /*-- ... while if an ALPHA step produces mediocre results, accept it */
    } else {
      p[ihi] = pr;
      y[ihi] = ypr;
    }

    /*-- go back for one more iteration */
  }
}


//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------


SlidingBumps::SlidingBumps(const PacVector<int>* adsN,  PacElemMultipole** adsP, PacTwissData* adsTwiss,
                           const PacVector<int>* dtsN,  PAC::Position* dtsP, PacTwissData* dtsTwiss, 
                           int plane, double mu, CorrectionRegime regime) {

    int i,j;
  
   _plane = plane;   // 0 -horizontal ; 1-vertical
    _regime = regime;  

   _correctors.latticeIndex = (vector<int> *) adsN;
   _correctors.strength0 = adsP;
   _correctors.twiss = adsTwiss;
   _correctors.corrNumber = adsN->size();
   _correctors.strength = new double[_correctors.corrNumber];

   for(i=0; i<_correctors.corrNumber; i++) _correctors.strength[i] = 0.;


   _monitors.latticeIndex = (vector<int> *) dtsN;
   _monitors.position = dtsP;
   _monitors.twiss = dtsTwiss;
   _monitors.monitNumber = dtsN->size();
   _monitors.status = new int[ _monitors.monitNumber];

   _mtb[0] = new int[ _monitors.monitNumber];
   _mtb[1] = new int[ _monitors.monitNumber];

   for(i=0; i<_monitors.monitNumber; i++)
     _monitors.status[i] = 0; // all monitors are initially turned off
     
   
   //bump constructing block
   if(_correctors.corrNumber  > 3) {
     _nBumps = 0;
     ThreeBump* bump_tmp = new ThreeBump[_correctors.corrNumber];
     u_short* creation_flag = new u_short[_correctors.corrNumber];
       
     for(i=0; i < _correctors.corrNumber; i++){
       creation_flag[i] = 0;
       if(!bump_tmp[i].ConstructBump(&_correctors, i, mu, plane)){
	 creation_flag[i] = 1;
	 _nBumps++;
       }
     }
     
     _bump = new ThreeBump[_nBumps]; 
     j=0;
     for(i=0; i < _correctors.corrNumber; i++){
       if(creation_flag[i]) _bump[j++] = bump_tmp[i];
     }
       
     delete [] bump_tmp;
     delete [] creation_flag;
     
     FindMonitors(mu);                          //    _mtb initialised here
   }
   else { 
     _bump = NULL;
     _nBumps = 0;
   }
}

//-----------------------------------------------------------------------------------------------

SlidingBumps::~SlidingBumps(){

       delete [] _bump; 
       delete _mtb[0];
       delete _mtb[1];
       delete [] _correctors.strength;
       delete []_monitors.status;
     }




//-----------------------------------------------------------------------------------------------

void SlidingBumps::FindMonitors(double mu){

// The algorithm here good for closed ring not for channel !! 


   int i, j, k;

// initialize monitor-bump table 
   for(j=0; j<2; j++) 
    for(i=0; i < _monitors.monitNumber; i++)
      _mtb[j][i] = -1;

// Find monitors inside bumps

  for(i=0; i < _nBumps; i++)
      _bump[i].FindMonitors(&_monitors, mu);

//  Make monitor-bump table for all bumps   
  for(j=0; j < _nBumps; j++) {  
     if(_regime == CO || !_bump[j].IfCrossed0()){
    for(k = 0; k < _bump[j]._numberOfMonitors; k++){  
          i = _bump[j]._monitIndex[k];
          if(_mtb[0][i] == -1) _mtb[0][i] = j;
          else if(j > _mtb[0][i]) _mtb[1][i] = j;
          else {_mtb[1][i] = _mtb[0][i]; _mtb[0][i] = j;}
       }
  }
	}

  // Make some additional stuff with monitor-bump table
  for(i=0; i < _monitors.monitNumber; i++) {
    if(_mtb[0][i] == 0 &&  _mtb[1][i] == _nBumps - 1){
      _mtb[0][i] = _mtb[1][i];
      _mtb[1][i] = 0 ;
    }  
  }
 }


//-----------------------------------------------------------------------------------------------


void SlidingBumps::SetMonitorStatus(int status, int inInd, int finInd){

    int i;

// Check if outside monitor range

    if(inInd < 0) inInd = 0;
    if(finInd > _monitors.monitNumber-1)  finInd = _monitors.monitNumber-1;



    for(i = inInd; i <= finInd; i++) 
        _monitors.status[i] = status;

  }


//-----------------------------------------------------------------------------------------------
const double PTOL = 1.0e-10;
const int ITMX1 = 50;

void SlidingBumps::CalculateCorrection(int fromDets, int toDets) {

  int i;

  // Check if outside monitor range
  if(fromDets < 0 ) fromDets  = 0;
  if(toDets > _monitors.monitNumber-1)  toDets = _monitors.monitNumber-1;

  // Check whether bumps have been constructed
  if(_nBumps == 0){
    cout << " Bumps have not been constructed yet. No orbit correction.\n";
    return;
  }

  //  Find which bump interval [inBump, finBump] would be used for correction
  int inBump;
  int finBump;

  while(_mtb[0][fromDets] == -1 &&  fromDets != toDets ){ 
    fromDets++;  
    if(fromDets == _monitors.monitNumber) fromDets = 0;
  }

  while(_mtb[1][toDets] == -1 &&  fromDets != toDets ) {
    if(_mtb[0][toDets] == -1){
      toDets--; 
      if(toDets == -1) toDets = _monitors.monitNumber-1;
    }
    else break;
  }
       
  if(fromDets == toDets && _mtb[0][fromDets] == -1){
    cout << "There are no bumps available for correction" << endl;
    return;      
  }

  inBump = _mtb[0][fromDets];
  finBump = (_mtb[1][toDets] == -1) ? _mtb[0][toDets] : _mtb[1][toDets];

  cout << "Correction from bump: " << inBump << "  to bump: " << finBump << 
    "  from detector: " << fromDets << "  to detector: " << toDets << endl;

  // Setup delta corrector strength equal to 0
  for(i=0; i<_correctors.corrNumber; i++) _correctors.strength[i] = 0.;   

  // Optimization loop
  float penold = 0.0;
  
  int iter = 0;
  while (1) {
    /*-- start of iteration loop */
    ++iter;
    float pentot = 0.0;

    int nOptimBumps;

    if(inBump <= finBump){
      nOptimBumps = finBump - inBump +1;
      for(i=inBump; i<=finBump; i++) {
	//      cout << "Opt1: bump= " << i << endl;
	pentot += _bump[i].Optimization(); }
    }
    else{
      nOptimBumps = _nBumps - finBump + inBump +1;
      for(i=inBump; i< _nBumps; i++) {
	pentot += _bump[i].Optimization();
      }
      for(i=0; i<=finBump; i++) {
	pentot += _bump[i].Optimization();
      }
    }
  
    pentot /= nOptimBumps;
    float tol = 2.0 * fabs( (pentot - penold) / (pentot + penold) );
    penold = pentot;

    if (iter>ITMX1) {
      /*-- Orbit solution didn't converge after ITMX1 iterations */
      printf(" ! ERROR ! Orbit non-convergent after %6d iters\n", ITMX1);
      break;
    } else if (tol>PTOL) {
      printf(" Finished iteration %2d with penalty % 15.10e...\n",
	     iter, pentot-1.);
      /*-- return for one more iteration */
    } else {
      printf(" Finished iteration %2d with penalty % 15.10e...\n",
	     iter, pentot-1.);  
      break; 
      /*-- within tolerance, so we're done */
    }
  }

  cout << "======================================================================" << endl;   
}
      
 
//-----------------------------------------------------------------------------------------------

void SlidingBumps::ApplyCorrection(){

   int i;

   if(_plane)  // Add to vertical correctors
      for(i=0; i<_correctors.corrNumber; i++)
          _correctors.strength0[i]->ktl(0) += _correctors.strength[i];
   else     // Add to horizontal correctors
       for(i=0; i<_correctors.corrNumber; i++)
          _correctors.strength0[i]->kl(0) -= _correctors.strength[i];

}

//-----------------------------------------------------------------------------------------------

void SlidingBumps::OpenLastBump(int det){

  cout << "det = " << det << endl;

     while(_mtb[0][det] == -1 &&  det != 0 ){ 
               det--;  
	     }
	     

  int bmp;

  if(_mtb[1][det] != -1)
      bmp = _mtb[1][det];
  else if(_mtb[0][det] != -1) 
      bmp = _mtb[0][det];  
  else { cout << "Can not open bump for detector: " << det << endl;
         return; }

  int monPos = (*_monitors.latticeIndex)[det];
  //  int i0 = _bump[bmp]._corrIndex[0];
  int i1 = _bump[bmp]._corrIndex[1];
  int i2 = _bump[bmp]._corrIndex[2]; 
  
  //   int corrPos0 = (*_correctors.latticeIndex)[i0];
  int corrPos1 = (*_correctors.latticeIndex)[i1];
  
  
     

   if(!_bump[bmp].IfCrossed0()) {  
      cout << "det = " << det << " det_mu= " << _monitors.twiss[det].mu(_plane) <<
      "   corr= " << i1 << "  corr_mu= " << _correctors.twiss[i1].mu(_plane)  <<  endl;   
      double sinph = fabs(_monitors.twiss[det].mu(_plane) - _correctors.twiss[i1].mu(_plane));
      _correctors.strength[i2] = 0.;  
      if(monPos < corrPos1 || sinph < 0.5)
            _correctors.strength[i1] = 0.;   
   }         
   

}

   
//-----------------------------------------------------------------------------------------------   

void  SlidingBumps::PrintBumps(ostream& out) {

   int i;

   

   for(i=0; i < _nBumps; i++) {
      out << "Bump " << i+1 << endl;
      _bump[i].Print(out);
  }

}


//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------

double GroteSteer::_pi = 3.14159265358979323846;

GroteSteer::GroteSteer(const PacVector<int>* adsN,  PacElemMultipole** adsP, PacTwissData* adsTwiss,
                           const PacVector<int>* dtsN,  PAC::Position* dtsP, PacTwissData* dtsTwiss, 
                           int plane) {

    int i;
  
   _plane = plane;   // 0 -horizontal ; 1-vertical

   _correctors.latticeIndex = (vector<int> *) adsN;
   _correctors.strength0 = adsP;
   _correctors.twiss = adsTwiss;
   _correctors.corrNumber = adsN->size();
   _correctors.strength = new double[_correctors.corrNumber];

   for(i=0; i<_correctors.corrNumber; i++) _correctors.strength[i] = 0.;


   _monitors.latticeIndex = (vector<int> *) dtsN;
   _monitors.position = dtsP;
   _monitors.twiss = dtsTwiss;
   _monitors.monitNumber = dtsN->size();
   _monitors.status = new int[ _monitors.monitNumber];

    for(i=0; i<_monitors.monitNumber; i++)
         _monitors.status[i] = 0;                  // all monitors are initially turned off
     
    corr2=_correctors.corrNumber-1;
    corr1=corr2;
   
}
//-----------------------------------------------------------------------------------------------

GroteSteer::~GroteSteer(){

       delete [] _correctors.strength;
       delete []_monitors.status;
     }



//-----------------------------------------------------------------------------------------------

const float PHASE_LIMIT = 0.3;


void GroteSteer::CalculateCorrection(int toDets) {

  // int i;


// Check if outside monitor range

    if(toDets < 0 ) toDets  = 0;
    if(toDets > _monitors.monitNumber-1)  toDets = _monitors.monitNumber-1;



    float phase_dif;
    
    int det2 = toDets;
    //    int det2_index = (*_monitors.latticeIndex)[det2];
    
    int det1 = det2; 

    do{
    if(--det1 < 0) break;
    // phase_dif = remainder(_monitors.twiss[det2].mu(_plane) - _monitors.twiss[det1].mu(_plane), _pi /* M_PI */);
    phase_dif = fmod(_monitors.twiss[det2].mu(_plane) - _monitors.twiss[det1].mu(_plane), _pi /* M_PI */);
    phase_dif = fabs(phase_dif);
    }while((phase_dif < PHASE_LIMIT || phase_dif > (1.-PHASE_LIMIT)));
    
    if(det1<0) { cout << " Bad orbit from the start. det1" << endl;  exit(0); } 

    int det1_index = (*_monitors.latticeIndex)[det1];

 
// Find 2 correctors upstream of monitors for use in the correction

   corr2 = _correctors.corrNumber-1;
   while(corr2 >= 0  && (*_correctors.latticeIndex)[corr2] > det1_index ) corr2--;
    
   if(corr2 < 1) { cout << " Bad orbit from the start. corr2" << endl;  exit(0); } 
     
    corr1 = corr2; 

 do{
    if(--corr1 < 0) break;
    // phase_dif = remainder(_correctors.twiss[corr2].mu(_plane) - _correctors.twiss[corr1].mu(_plane), _pi /* M_PI */);
    phase_dif = fmod(_correctors.twiss[corr2].mu(_plane) - _correctors.twiss[corr1].mu(_plane), _pi /* M_PI */);
    phase_dif = fabs(phase_dif);
    }while(phase_dif < PHASE_LIMIT || phase_dif > (1.-PHASE_LIMIT));
    
    if(corr1<0) { cout << " Bad orbit from the start. corr1" << endl;  exit(0); }     


 cout << " Monitors used: det1= " << det1 << " ("<< (*_monitors.latticeIndex)[det1] << ")" <<
         "    det2= " << det2 << " ("<< (*_monitors.latticeIndex)[det2] << ")" << endl;
 cout << " Correctors used: corr1= " << corr1 << " ("<< (*_correctors.latticeIndex)[corr1] << ")" <<
         "    corr2= " << corr2 << " ("<< (*_correctors.latticeIndex)[corr2] << ")" << endl;

 
// Calculate responce coefficients between correctors and monitors

float bet_m1 = _monitors.twiss[det1].beta(_plane);
float bet_m2 = _monitors.twiss[det2].beta(_plane);
float mu_m1 = _monitors.twiss[det1].mu(_plane);
float mu_m2 = _monitors.twiss[det2].mu(_plane);

float bet_c1 = _correctors.twiss[corr1].beta(_plane);
float bet_c2 = _correctors.twiss[corr2].beta(_plane);
float mu_c1 = _correctors.twiss[corr1].mu(_plane);
float mu_c2 = _correctors.twiss[corr2].mu(_plane);


float a11 = sqrt(bet_m1*bet_c1)*sin(mu_m1-mu_c1);     
float a12 = sqrt(bet_m1*bet_c2)*sin(mu_m1-mu_c2); 
float a21 = sqrt(bet_m2*bet_c1)*sin(mu_m2-mu_c1); 
float a22 = sqrt(bet_m2*bet_c2)*sin(mu_m2-mu_c2); 

float determ = a22*a11-a12*a21;

cout << "a11= " << a11 << " a12= " << a12 << " a21= " << a21 << " a22= " << a22 <<" det=" << determ << endl;

if(fabs(determ) < 0.1)  cout << "Warning: small determinant" << endl;
 
 
float posit1 = (_monitors.position[det1])[2*_plane];
float posit2 = (_monitors.position[det2])[2*_plane];

 cout << " Positions on monitors: on det1= " << posit1 <<
         "   on det2= " << posit2 << endl;  

_correctors.strength[corr1] = (a12*posit2-a22*posit1)/determ;
_correctors.strength[corr2] = (a21*posit1-a11*posit2)/determ;

cout << " Corrector strength: c1= " << _correctors.strength[corr1] << 
      "   c2= " << _correctors.strength[corr2] << endl;


}
      
 
//-----------------------------------------------------------------------------------------------

void GroteSteer::ApplyCorrection(){

  //   int i;

   if(_plane){  // Add to vertical correctors
          _correctors.strength0[corr1]->ktl(0) += _correctors.strength[corr1];
          _correctors.strength0[corr2]->ktl(0) += _correctors.strength[corr2];
        }
   else {    // Add to horizontal correctors
          _correctors.strength0[corr1]->kl(0) -= _correctors.strength[corr1];
          _correctors.strength0[corr2]->kl(0) -= _correctors.strength[corr2];
        }
}
