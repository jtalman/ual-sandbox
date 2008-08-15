#ifndef SLIDING_BUMPS
#define SLIDING_BUMPS

#include "Main/TeapotMatrixService.h"
#include "Main/TeapotTwissService.h"
#include "Main/Teapot.h"

#include <iostream>

enum CorrectionRegime {FT, CO};    
// enum Boolean {false, true};      






struct CorrectorSet {
   PacVector<int>* latticeIndex;
   PacElemMultipole**strength0;
   double*   strength; 
   PacTwissData* twiss;
   int corrNumber;

   void Print(ostream& out=cout, int plane=0);
 };

struct MonitorSet {
   PacVector<int>* latticeIndex;
   PAC::Position* position;
   PacTwissData* twiss;
   int monitNumber;
   int* status;

  void Print(ostream& out=cout, int plane=0);
 };








class ThreeBump {

 friend class SlidingBumps;

private:   int _plane;                   // 0 - horizontal. 1 -vertical 

           CorrectorSet* _correctors;    // pointer on outside set of correctors
           int _corrIndex[3];            // indexes of bump correctors in the above corrector set
           float  _correctorResponce[3]; // to produce 1mm orbit at central corrector
           
           MonitorSet* _monitors;         // pointer on outside set of monitors
           int* _monitIndex;             // indexes of bump monitors in the above monitor set
           int _numberOfMonitors;        // number of monitors in the bump
           float* _monitorResponce;     //  monitor data when central corrector has 1mm orbit

// because of CC compiler failed to use function pointer I excluded function
// pointer from 'onedim' argumets. 'PenaltyFunction' now is inside 'onedim' body
           
           double PenaltyFunction(double strbmp);   // descriprion of penalty function to optimize
           void onedim(double *psol, int *itmx, int itw, int iprnt,
	               double pfin, double ftol);               // optimization routine

public:  ThreeBump();
         ThreeBump(CorrectorSet* , int middleCorrIndex, double mu_tot, int plane); 
         ~ThreeBump();

         ThreeBump& operator= (const ThreeBump&); 

// find corrector relation to form close 3-bump
         int ConstructBump(CorrectorSet* , int middleCorrIndex, double mu_tot, int plane);  
               
         void FindMonitors(MonitorSet*, double mu );    // find which monitors are inside the bump  
        
         double Optimization();        // minimize penalty function for given bump

         void Print(ostream& out = cout);            // print bump configuration to cout
         void PrintCorrector(ostream& out = cout, int i=0);
         void PrintMonitor(ostream& out = cout, int i=0);

         // Boolean IfCrossed0(void);  // new add  
	 bool IfCrossed0(void);  // new add  
       
};



class GroteSteer {

 private:
            CorrectorSet _correctors;
            MonitorSet   _monitors; 
   
            int corr1, corr2;  // Indexes of correctors used in correction
            int _plane;

	    static double _pi;

  public:  GroteSteer(const PacVector<int>* adsN,  PacElemMultipole** adsP, PacTwissData* adsTwiss,
                      const PacVector<int>* dtsN,  PAC::Position* dtsP, PacTwissData* dtsTwiss, 
                      int plane);
         
            ~GroteSteer();   
   
            void CalculateCorrection(int toDets);  // Make correction to toDets monitor
            void ApplyCorrection();                // Write corrector set to TPOT PacElemMultipole data

	  };


class SlidingBumps {

private: ThreeBump* _bump;
         int _nBumps;

         CorrectorSet _correctors;
         MonitorSet   _monitors; 
    
         int* _mtb[2];          // _mtb[0][i], _mtb[1][i] shows the bump numbers for i-th monitor (-1 if
				// the monitor is inactive inside the bump)					   
         int _plane;

         CorrectionRegime  _regime;
   
public:   
         SlidingBumps(const PacVector<int>* adsN,  PacElemMultipole** adsP, PacTwissData* adsTwiss,
                      const PacVector<int>* dtsN,  PAC::Position* dtsP, PacTwissData* dtsTwiss, 
                      int plane,double mu, CorrectionRegime regime = CO);   
         
         ~SlidingBumps();         

         void FindMonitors(double mu);    // find monitors inside bumps; fill _mtb (monitor-bump table)

// set status of monitors inside the range [inInd,finInd]
         void SetMonitorStatus(int status, int inInd, int finInd);   

// set status of all monitors
void SetMonitorStatus(int status){SetMonitorStatus(status, 0, _monitors.monitNumber-1);};  // new add

        // Make correction between fromDets and toDets monitors
         void CalculateCorrection(int fromDets, int toDets);  
         void ApplyCorrection();                                // Write corrector set to TPOT PacElemMultipole data

         void OpenLastBump(int det);              // Open last bump containing detectors det

         void PrintBumps(ostream& out = cout);                      // Print bumps configuration to cout

         void PrintCorrectors(ostream& out = cout);    // Print correctors  to out
         void PrintMonitors(ostream& out = cout);  
};

inline void SlidingBumps::PrintCorrectors(ostream& out) {   _correctors.Print(out,_plane);}  
inline void SlidingBumps::PrintMonitors(ostream& out)  {    _monitors.Print(out,_plane);}


#endif
 
          
