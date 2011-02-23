#include <iostream>
#include <fstream>
#include <iomanip>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"

#include "SPINK/Propagator/DipoleTracker.hh"
#include "SPINK/Propagator/RFCavityTracker.hh"
#include "SPINK/Propagator/SpinTrackerWriter.hh"
#include "SPINK/Propagator/SnakeTransform.hh"
#include "SPINK/Propagator/RFSolenoid.hh"

#include "timer.h"
#include "PositionPrinter.h"
#include "SpinPrinter.h"

using namespace UAL;

int main(){

  UAL::Shell shell;

  double cc  = 2.99792458E+8;
  double G = 1.7928456;
  double mass   = 0.938272029;            //       proton mass [GeV]
  double charge = 1.0;

  /** AUL:17MAR10 _____________________________________________________________________*/
  /**********************************************************/
  //* Read input parameters*/
  /**********************************************************/
 
  std::ifstream configInput("./data/spink.in");//AULNLD:07JAN10

  std::string dummy ; // this string has been added to improve readability of input
  std::string variantName;
  bool outdmp;
  bool logdmp;
  int irSBend; int irQuad;
  double gamma;
  double tuneX, tuneY, chromX, chromY;
  bool settunes; //AUL:08MAR10
  double dgammadt; //double dgammadt = 1.1522754; // 1/s
  double V; double harmon ; //V = 1.5e-04, harmon = 0
  double ssx; double ssy; double ssz;
  double emit_y; // Pi mm*mrad (normalized) 15.; 
  double emit_x; // Pi mm*mrad (normalized)
  double x00; double x00p; double y00; double y00p; double ct0; double dpp0;
  bool calcPhaseSpace;
  bool snkflag ; //AUL:10MAR10
  double mu1; double mu2; double phi1; double phi2; double the1; double the2;
  int turns;
  double RFS_Bdl;
  char RFS_rot;
  double RFS_freq0;
  double RFS_dfreq;
  int RFS_nt; 

  configInput >> dummy >> variantName;
  configInput >> dummy >> outdmp ; //AUL:12MAR10
  configInput >> dummy >> logdmp ; //std::cout << "logdmp=" << logdmp << std::endl;
  configInput >> dummy >> irSBend >> irQuad;
  configInput >> dummy >> gamma; 
  configInput >> dummy >> tuneX >> tuneY ;
  configInput >> dummy >> chromX >> chromY;
  configInput >> dummy >> settunes ; //AUL:08MAR10
  configInput >> dummy >> dgammadt;
  configInput >> dummy >> V >> harmon;
  configInput >> dummy >> ssx >> ssy >> ssz; 
  configInput >> dummy >> emit_x >> emit_y;
  configInput >> dummy >> x00 >> x00p >> y00 >> y00p >> ct0 >> dpp0;
  configInput >> dummy >> calcPhaseSpace; 
  configInput >> dummy >> snkflag; 
  configInput >> dummy >> mu1 >> mu2 ; 
  configInput >> dummy >> phi1 >> phi2 ; 
  configInput >> dummy >> the1 >> the2 ;
  configInput >> dummy >> turns;
  configInput >> dummy >> RFS_Bdl;
  configInput >> dummy >> RFS_rot;
  configInput >> dummy >> RFS_freq0;
  configInput >> dummy >> RFS_dfreq;
  configInput >> dummy >> RFS_nt; 
  /** AUL:17MAR10 _________________________________________________________________*/

    std::cout <<  variantName << std::endl;
    std::cout <<  outdmp << std::endl;
    std::cout <<  logdmp << std::endl;
    std::cout <<  irSBend << " " << irQuad << std::endl;
    std::cout <<  gamma << std::endl;
    std::cout <<  tuneX << " " << tuneY << std::endl;
    std::cout <<  chromX << " " << chromY << std::endl;
    std::cout <<  settunes  << std::endl;
    std::cout <<  dgammadt << std::endl;
    std::cout <<  V << " " << harmon << std::endl;
    std::cout <<  ssx << " " << ssy << " " << ssz << std::endl;
    std::cout <<  emit_x << " " << emit_y << std::endl;
    std::cout <<  x00 << " " <<x00p << " " << y00 << " " << y00p << " " << ct0 << " " << dpp0 << std::endl;
    std::cout <<  calcPhaseSpace  << std::endl;
    std::cout <<  snkflag  << std::endl;
    std::cout <<  mu1 << " " << mu2 << std::endl;
    std::cout <<  phi1 << " " << phi2 << std::endl;
    std::cout <<  the1 << " " << the2 << std::endl;
    std::cout <<  turns << std::endl;
    std::cout <<  RFS_Bdl << std::endl;
    std::cout <<  RFS_rot << std::endl;
    std::cout <<  RFS_freq0 << std::endl;
    std::cout <<  RFS_dfreq << std::endl;
    std::cout <<  RFS_nt << std::endl;

  // ************************************************************************

  SPINK::SnakeTransform::setOutputDump(outdmp); //AUL:01MAR10
  SPINK::RFSolenoid::setOutputDump(outdmp); //AUL:01MAR10
  SPINK::DipoleTracker::setOutputDump(outdmp); //AUL:02MAR10
  SPINK::RFCavityTracker::setOutputDump(outdmp); //AUL:27APR10

  // ************************************************************************
  if( logdmp ){std::cout << "\nDefine the space of Taylor maps." << std::endl;}
  // ************************************************************************

  shell.setMapAttributes(Args() << Arg("order", 5));

  // ************************************************************************
  if( logdmp ){  std::cout << "\nBuild lattice." << std::endl;}
  // ************************************************************************

  std::string sxfFile = "./data/";
  sxfFile += variantName;
  sxfFile += ".sxf";

  std::cout << "sxfFile = " << sxfFile << endl;

  shell.readSXF(Args() << Arg("file",  sxfFile.c_str()));

  // ************************************************************************
  if( logdmp ){  std::cout << "\nAdd split ." << std::endl;}
  // ************************************************************************

  if( logdmp ){std::cout << "irSBend = " << irSBend << ", irQuad = " << irQuad << endl;}

  shell.addSplit(Args() << Arg("lattice", "ring") << Arg("types", "Sbend")
  		 << Arg("ir", irSBend));

  shell.addSplit(Args() << Arg("lattice", "ring") << Arg("types", "Quadrupole")
  		 << Arg("ir", irQuad));

  // ************************************************************************
  if( logdmp ){  std::cout << "Select lattice." << std::endl;}
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "ring"));

  // ************************************************************************
  if( logdmp ){  std::cout << "\nWrite ADXF file ." << std::endl;}
  // ************************************************************************

  std::string outputFile = "./out/cpp/";
  outputFile += variantName;
  outputFile += ".sxf";

  shell.writeSXF(Args() << Arg("file",  outputFile.c_str()));

  // ************************************************************************
  if( logdmp ){std::cout << "\nDefine beam parameters." << std::endl;}
  // ************************************************************************

  double energy = gamma*mass;

  shell.setBeamAttributes(Args() << Arg("energy", energy) << Arg("mass", mass)
			  << Arg("charge",charge));

  PAC::BeamAttributes& ba = shell.getBeamAttributes();

  // ************************************************************************
  if( logdmp ){  std::cout << "\nLinear analysis." << std::endl;}
  // ************************************************************************
  
  // Make linear matrix

  std::string mapFile = "./out/cpp/";
  mapFile += variantName;
  mapFile += ".map1";

  if( logdmp ){  std::cout << " matrix" << std::endl;}
  shell.map(Args() << Arg("order", 1) << Arg("print", mapFile.c_str()));

  // ************************************************************************
  if( logdmp ){  std::cout << "\nTune and chromaticity fitting. " << std::endl;}
  // ************************************************************************

  // shell.analysis(Args());

  // for RHIC AUL:07MAY10 after a hint by Nikolay 
  if( settunes ){
    shell.tunefit(Args() << Arg("tunex", tuneX) << Arg("tuney", tuneY) << Arg("b1f", "^qf$") << Arg("b1d", "^qd$"));
    shell.chromfit(Args() << Arg("chromx", chromX) << Arg("chromy", chromY)<< Arg("b2f", "^sf$") << Arg("b2d", "^sd$"));
  } else {
    if( logdmp ){ std::cout << "\n--tunes and chromaticity NOT readjusted" << std::endl;}
  }
   
  /* for EDM AUL:07MAY10 after a hint by Nikolay
  if( settunes ){
    shell.tunefit(Args() << Arg("tunex", tuneX) << Arg("tuney", tuneY) << Arg("b1f", "^quadf$") << Arg("b1d", "^quadd$"));
    shell.chromfit(Args() << Arg("chromx", chromX) << Arg("chromy", chromY)<< Arg("b2f", "^sexf$") << Arg("b2d", "^sexd$"));
  } else {
    if( logdmp ){ std::cout << "\n--tunes and chromaticity NOT readjusted" << std::endl;}
  }
  */

  /* for SCT AUL:12MAY10 after a hint by Nikolay */
  /*
  if( settunes ){
    shell.tunefit(Args() << Arg("tunex", tuneX) << Arg("tuney", tuneY) << Arg("b1f", "^quada$") << Arg("b1d", "^quadb$"));
    shell.chromfit(Args() << Arg("chromx", chromX) << Arg("chromy", chromY)<< Arg("b2f", "^sexta$") << Arg("b2d", "^sextb$"));
  } else {
    if( logdmp ){ std::cout << "\n--tunes and chromaticity NOT readjusted" << std::endl;}
  }
*/
  // Calculate twiss
  
  std::string twissFile = "./out/cpp/";
  twissFile += variantName;
  twissFile += ".twiss";

  if( logdmp ){  std::cout << " twiss " << std::endl;}

  //AUL shell.twiss(Args() << Arg("print", twissFile.c_str()));

  std::cout << " calculate suml" << std::endl;
  //AUL  shell.analysis(Args());

  // ************************************************************************
  std::cout << "\nAlgorithm Part. " << std::endl;
  // ************************************************************************

  std::string apdfFile = "./data/spink.apdf";

  UAL::APDF_Builder apBuilder;

  apBuilder.setBeamAttributes(ba);

  UAL::AcceleratorPropagator* ap = apBuilder.parse(apdfFile);

  if(ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return 1;
  }
  if( logdmp ){
    std::cout << "\nSpink tracker, ";
    std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

    // ************************************************************************
    std::cout << "\nSet Acceleration. " << std::endl;
    // ************************************************************************
  }
  double dedt = dgammadt*mass;
  double circum = OpticsCalculator::getInstance().suml; 
  double T_0 = circum / cc;

  double lag = asin((dedt * T_0)/(2*V))/(2*UAL::pi);

  if( logdmp ){
    cout << "dgamma/dt = " << dgammadt << endl ; //AUL:29DEC09
    cout << "Circumference(m) = " << circum << endl;
    cout << "Volt = " << V << ", harmon =" << harmon << ", lag = " << lag << std::endl;
  }

  SPINK::RFCavityTracker::setRF(V, harmon, lag);  //AUL:17MAR10
  //double circ = circum;
  SPINK::RFCavityTracker::setCircum(circum); //AUL:17MAR10

  // ************************************************************************
  if( logdmp ){  std::cout << "\nBunch Part." << std::endl;}
  // ************************************************************************

  ba.setG(G);         // proton G factor
  
  if( logdmp ){  cout << "gamma = " << gamma << ",  Ggamma = " << G*gamma << endl;}

  PAC::Bunch bunch(1);               // bunch with one particle
  bunch.setBeamAttributes(ba);

  if( logdmp ){  std::cout << "initial spin = " << ssx << "  " << ssy << "  " << ssz << std::endl;}

  PAC::Spin spin;
  spin.setSX(ssx);
  spin.setSY(ssy);
  spin.setSZ(ssz);

  //double amplit_y = 15.; // Pi mm*mrad (normalized) 15.; 
  //double amplit_x = 0.; // Pi mm*mrad (normalized)
  //double dpp0 = 0.0;
  
  double x0; double x0p; double y0; double y0p;

  if( logdmp ){
    std::cout << "emit_x = " << emit_x << ", emit_y = " << emit_y << std::endl; //AUL:30DEC09

    // ************************************************************************
    std::cout << "\nOptics" << std::endl; //AUL:30DEC09
    // ************************************************************************
  }
  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  //AUL  optics.calculate();

  PacTwissData tws = optics.m_chrom->twiss();
  double q_x = tws.mu(0)/2./UAL::pi;
  double q_y = tws.mu(1)/2./UAL::pi;
  double beta_x = tws.beta(0);
  double beta_y = tws.beta(1);
  double chrm_x = optics.m_chrom->dmu(0)/2./UAL::pi;
  double chrm_y = optics.m_chrom->dmu(0)/2./UAL::pi;
  double alfa_x = tws.alpha(0);
  double alfa_y = tws.alpha(1);

  if( logdmp ){
    std::cout << "beta_x = " << beta_x << "  beta_y = " << beta_y << std::endl;
    std::cout << "alfa_x = " << alfa_x << "  alfa_y = " << alfa_y << std::endl;
    std::cout << "Q_x = " << q_x << "  Q_y = " << q_y << std::endl;
    std::cout << "chrom_x = " << chrm_x << "  chrom_y = " << chrm_y << std::endl;
  }

  if( calcPhaseSpace){
 
    if( logdmp ){ std::cout << "\nTranverse phase space calculated from emittance" << endl;}
    
    x0 = sqrt(emit_x*beta_x/(6*gamma))*0.001 + tws.d(0)*dpp0;
    x0p = tws.dp(0)*dpp0;
    y0 = sqrt(emit_y*beta_y/(6*gamma))*0.001 + tws.d(1)*dpp0;
    y0p = tws.dp(1)*dpp0;

  } else {

    if( logdmp ){ std::cout << "\nTranverse phase space directly input" << endl;}

    x0 = x00 + tws.d(0)*dpp0;
    x0p = x00p + tws.dp(0)*dpp0;
    y0 = y00 + tws.d(1)*dpp0;
    y0p = y00p + tws.dp(1)*dpp0;
  }

  if( logdmp ){
    std::cout << "\nInitial phase space (including dispersion)" << std::endl; //AUL:17MAR10
    std::cout << "x0 = " << x0 << ",  x0p = " << x0p << std::endl;
    std::cout << "y0 = " << y0 << ",  y0p = " << y0p << std::endl;
    std::cout << "ct0 = " << ct0 << ",  dpp0 = " << dpp0 << std::endl; //AUL:17MAR10
  }

  for(int ip=0; ip < bunch.size(); ip ++){
    bunch[ip].getPosition().set(x0, x0p, y0, y0p, ct0, dpp0);    //AUL:17MAR10
    bunch[ip].setSpin(spin);
  }

  /** read in snake parameters AULNLD 2/9/10 */

  if( snkflag ){  //AUL:07MAY10

    SPINK::SnakeTransform::setSnakeParams(mu1, mu2, phi1, phi2, the1, the2);

    if( logdmp ){    
      std::cout << "\nSnakes " << std::endl;
      std::cout << "snk1_mu = " << mu1 << ", snk2_mu = " << mu2 << endl;
      std::cout << "snk1_phi = " << phi1 << ", snk2_phi = " << phi2 << endl;
      std::cout << "snk1_theta = " << the1 << ", snk2_theta = " << the2 << endl;
    }
  }
  else 
  {
      if( logdmp ){
	std::cout << "\nNo Snakes" << std::endl ;
      }
  }

  SPINK::RFSolenoid::setRFSParams(RFS_Bdl, RFS_rot, RFS_freq0, RFS_dfreq, RFS_nt);

  // ************************************************************************
  if( logdmp ){  std::cout << "\nTracking. " << std::endl;}
  // ************************************************************************

  double t; // time variable
  
  //  if( logdmp ){ std::cout << "\nTurns = " << turns << std::endl ;}
  std::cout << "\nTurns = " << turns << std::endl ;
  //return 0;

  std::string orbitFile = "./out/cpp/";
  orbitFile += variantName;
  orbitFile += ".orbit";

  PositionPrinter positionPrinter;
  positionPrinter.open(orbitFile.c_str());

  std::string spinFile = "./out/cpp/";
  spinFile += variantName;
  spinFile += ".spin";
  
  SpinPrinter spinPrinter;
  spinPrinter.open(spinFile.c_str());

  ba.setElapsedTime(0.0);

  start_ms();

  for(int iturn = 1; iturn <= turns; iturn++){

    /** to pass turn no for diagnostics AUL:02MAR10 */
    SPINK::SnakeTransform::setNturns(iturn);
    SPINK::DipoleTracker::setNturns(iturn);
    SPINK::RFCavityTracker::setNturns(iturn);//AUL:27APR10

    ap -> propagate(bunch);
    
    for(int ip=0; ip < bunch.size(); ip++){
       positionPrinter.write(iturn, ip, bunch);
       spinPrinter.write(iturn, ip, bunch);
    }

  }

  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;

  positionPrinter.close();
  spinPrinter.close();

  return 1;
}

