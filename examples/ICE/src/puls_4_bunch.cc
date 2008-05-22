
#include <iostream>

#include "UAL/Common/Def.hh"
#include "PAC/Beam/Bunch.hh"
#include "ICE/TImpedance/TImpedanceWF.hh"

int main()
{

  // ************************************************************************
  std::cout << "\nBeam parameters \n" << std::endl;
  // ************************************************************************

  int nparts = 10000;

  PAC::Bunch bunch(nparts);

  bunch.setEnergy(1.93827231);
  bunch.setMacrosize(2.084e14/nparts);

  double e     = bunch.getEnergy();
  double m     = bunch.getMass();
  double beta  = sqrt(e*e - m*m)/e;
  double gamma = 1./sqrt(1. - beta*beta);

  std::cout <<  "energy     = " << e                     << std::endl;
  std::cout <<  "mass       = " << m                     << std::endl;
  std::cout <<  "beta       = " << beta                  << std::endl;
  std::cout <<  "gamma      = " << gamma                 << std::endl;
  std::cout <<  "clight     = " << UAL::clight           << std::endl;
  std::cout <<  "bunchsize  = " << bunch.size()          << std::endl;
  std::cout <<  "macrosize  = " << bunch.getMacrosize()  << std::endl;

  // ************************************************************************
  std::cout << "\nLattice parameters \n" << std::endl;
  // ************************************************************************

  double ringLength   = 220.;
  double rfNumber     = 1.;
  double revFrequency = rfNumber*(beta*UAL::clight/ringLength);
  double revTime      = 1./revFrequency;

  std::cout <<  "ringLength = " << ringLength  << std::endl; 
  std::cout <<  "revTime    = " << revTime     << std::endl;

  // ************************************************************************
  std::cout << "\nBunch distribution \n" << std::endl;
  // ************************************************************************

  double x_disp   = 1.0;
  double y_disp   = 1.0;
  double time_max =   revTime/4.;
  double time_min = - revTime/4.;

  std::cout <<  "x_disp     = " << x_disp      << std::endl;   
  std::cout <<  "y_disp     = " << y_disp      << std::endl;   
  std::cout <<  "time_max   = " << time_max    << std::endl;   
  std::cout <<  "time_min   = " << time_min    << std::endl;   


  double xx, yy, ct;
  double ct_step = UAL::clight*(time_max - time_min)/nparts;
  for (int ip = 0; ip < nparts; ip++){
     ct = ip*ct_step + UAL::clight*time_min;
     bunch[ip].getPosition().set(x_disp, 0.0, y_disp, 0.0, ct, 0.0); 
   } 

  // ************************************************************************
  std::cout << "\nResonant element parameters \n" << std::endl;
  // ************************************************************************

  double Rs_x    = 1000;
  double Rs_y    = 1000;
  double Qfact_x = 100;
  double Qfact_y = 100;
  double rsFrq_x = 5*revFrequency;
  double rsFrq_y = 5*revFrequency;


  std::cout << "resistance parameter (x) [ohms/m*m]   = " << Rs_x     << std::endl;   
  std::cout << "resistance parameter (y) [ohms/m*m]   = " << Rs_y     << std::endl;
  std::cout << "quality factor (x)                    = " << Qfact_x  << std::endl;
  std::cout << "quality factor (y)                    = " << Qfact_y  << std::endl;
  std::cout << "resonant frequency (x)  [Hz]          = " << rsFrq_x  << std::endl;
  std::cout << "resonant frequency (y)  [Hz]          = " << rsFrq_y  << std::endl;

  // ************************************************************************
  std::cout << "\nTImpedance element \n" << std::endl;
  // ************************************************************************

  int nBuns = 100;
  int n_max_elements = 3 ;
  int max_bunch_size = 100000;

  std::cout << "Grid size          = " << nBuns           << std::endl;
  std::cout << "n_max_elements     = " << n_max_elements  << std::endl;
  std::cout << "Max bunch size     = " << max_bunch_size  << std::endl;

  ICE::TImpedanceWF tImpEl(nBuns, n_max_elements, max_bunch_size);

  tImpEl.addResonantElement(0, Rs_x, Qfact_x, rsFrq_x);
  tImpEl.addResonantElement(1, Rs_y, Qfact_y, rsFrq_y);

  // ************************************************************************
  std::cout << "\nTracking \n" << std::endl;
  // ************************************************************************

  std::cout <<  "turn 0" << std::endl;
  tImpEl.propagate(bunch , 0.0);

  std::cout <<  "turn 1" << std::endl;
  tImpEl.propagate(bunch, revTime);

  std::cout <<  "turn 2" << std::endl;
  tImpEl.propagate(bunch, revTime);

  std::cout <<  "turn 3" << std::endl;
  tImpEl.propagate(bunch, revTime);

  std::cout <<  "turn 4" << std::endl;
  tImpEl.propagate(bunch, revTime);


  tImpEl.showXY("./out/xy_puls_4_bunch.dat");
  tImpEl.showTKick("./out/TKick_puls_4_bunch.dat");


  return 1;

}
