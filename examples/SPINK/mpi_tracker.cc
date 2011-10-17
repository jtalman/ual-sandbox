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
//#include "TEAPOT/Integrator/RFCavityTracker.hh"
#include "SPINK/Propagator/RFCavityTracker.hh"
#include "SPINK/Propagator/SpinTrackerWriter.hh"
#include "SPINK/Propagator/SnakeTransform.hh"

#include "timer.h"
#include "PositionPrinter.h"
#include "SpinPrinter.h"
#include "mpi.h"


using namespace UAL;

int main(int argc, char *argv[]){
//int main() {
  int numprocs, myrank;


  UAL::Shell shell;

  double cc  = 2.99792458E+8;
  double G = 1.7928456;
  double mass   = 0.938272029;            //       proton mass [GeV]
  double charge = 1.0;
  double X[100],PX[100],Y[100],PY[100],CT[100],DP[100];
  double SX[100],SY[100],SZ[100];

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // MPI_Status status;
    //char greeting[MPI_MAX_PROCESSOR_NAME + 80];

  if(myrank == 0){
   std::cout << "in head node with numprocs =" << numprocs << "\n";}
 
  /** AUL:17MAR10 _____________________________________________________________________*/
  /**********************************************************/
  //* Read input parameters*/
  /**********************************************************/
 
  std::ifstream configInput("./data2/spinkE.in");//AULNLD:07JAN10

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
  int calcPhaseSpace;
  bool snkflag ; //AUL:10MAR10
  double mu1; double mu2; double phi1; double phi2; double the1; double the2;
  int turns;

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


  std::cout << "Turns = " << turns << " \n";
  std::cout << "the2 =" << the2 << "\n";
  /** AUL:17MAR10 _________________________________________________________________*/

 
  // ************************************************************************

  SPINK::SnakeTransform::setOutputDump(outdmp); //AUL:01MAR10
  SPINK::DipoleTracker::setOutputDump(outdmp); //AUL:02MAR10
  SPINK::RFCavityTracker::setOutputDump(outdmp); //AUL:27APR10
  
  // ************************************************************************
  if( logdmp ){std::cout << "\nDefine the space of Taylor maps." << std::endl;}
  // ************************************************************************

  shell.setMapAttributes(Args() << Arg("order", 5));

  // ************************************************************************
  if( logdmp ){  std::cout << "\nBuild lattice." << std::endl;}
  // ************************************************************************

  std::string sxfFile = "./data2/";
  char Crank[3];
  sxfFile += variantName;
  sprintf(Crank, "%d",myrank);
  sxfFile += ".sxf";

  std::cout << "sxfFile = " << sxfFile << endl;

  // if(myrank == 0){
  shell.readSXF(Args() << Arg("file",  sxfFile.c_str()));

  // ************************************************************************
  if( logdmp ){  std::cout << "\nAdd split ." << std::endl;}
  // ************************************************************************

  if( logdmp ){std::cout << "irSBend = " << irSBend << ", irQuad = " << irQuad << endl;}

  shell.addSplit(Args() << Arg("lattice", "rhic") << Arg("types", "Sbend")
  		 << Arg("ir", irSBend));

  shell.addSplit(Args() << Arg("lattice", "rhic") << Arg("types", "Quadrupole")
  		 << Arg("ir", irQuad));

  // ************************************************************************
  if( logdmp ){  std::cout << "Select lattice." << std::endl;}
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "rhic"));

  // ************************************************************************
  if( logdmp ){  std::cout << "\nWrite ADXF file ." << std::endl;}
  // ************************************************************************

  std::string outputFile = "./EY20EX800_out/";
  outputFile += variantName;
  outputFile += Crank;
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

  std::string mapFile = "./EY20EX800_out/";
  mapFile += variantName;
  mapFile += Crank;
  mapFile += ".map1";

  if( logdmp ){  std::cout << " matrix" << std::endl;}
  shell.map(Args() << Arg("order", 1) << Arg("print", mapFile.c_str()));

  // ************************************************************************
  if( logdmp ){  std::cout << "\nTune and chromaticity fitting. " << std::endl;}
  // ************************************************************************

  // shell.analysis(Args());

  /* for RHIC AUL:07MAY10 after a hint by Nikolay */
  if( settunes ){
    shell.tunefit(Args() << Arg("tunex", tuneX) << Arg("tuney", tuneY) << Arg("b1f", "^qf$") << Arg("b1d", "^qd$"));
      shell.chromfit(Args() << Arg("chromx", chromX) << Arg("chromy", chromY)<< Arg("b2f", "^sf") << Arg("b2d", "^sd"));
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
  // if( settunes ){
  // shell.tunefit(Args() << Arg("tunex", tuneX) << Arg("tuney", tuneY) << Arg("b1f", "^quada$") << Arg("b1d", "^quadb$"));
  // shell.chromfit(Args() << Arg("chromx", chromX) << Arg("chromy", chromY)<< Arg("b2f", "^sexta$") << Arg("b2d", "^sextb$"));
  // } else {
  // if( logdmp ){ std::cout << "\n--tunes and chromaticity NOT readjusted" << std::endl;}
  //}

  // Calculate twiss
  
  std::string twissFile = "./EY20EX800_out/";
  twissFile += variantName;
  twissFile += Crank;
  twissFile += ".twiss";

  if( logdmp ){  std::cout << " twiss " << std::endl;}

  std::cout << "we are here \n";
  //shell.twiss(Args() << Arg("print", twissFile.c_str()));

  std::cout << "we are here next \n";
  std::cout << " calculate suml" << std::endl;
  shell.analysis(Args());

  // ************************************************************************
  std::cout << "\nAlgorithm Part. " << std::endl;
  // ************************************************************************

  std::string apdfFile = "./data2/spink.apdf";

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

  // dgammadt = 2.094  V = 1.3e-4 
  double dedt = dgammadt*mass;
  double circum = OpticsCalculator::getInstance().suml; 
  double T_0 = circum / cc;

    double lag = asin((dedt * T_0)/(V))/(2*UAL::pi);
    double offset = lag*circum/harmon;
  // double lag = asin(dedt*T_0/V);
    // lag = 0.0;
    if( logdmp ){ }
    cout << "dgamma/dt = " << dgammadt << endl ; //AUL:29DEC09
    cout << "Circumference(m) = " << circum << endl;
    cout << "Volt = " << V << ", harmon =" << harmon << ", lag = " << lag*360 << std::endl;
  

  SPINK::RFCavityTracker::setRF(V,120,lag);
  //  TEAPOT::RFCavityTracker  tracker;
  //tracker.setRF(V, harmon, lag);  //AUL:17MAR10
  //double circ = circum;
   SPINK::RFCavityTracker::setCircum(circum); //AUL:17MAR10

  // ************************************************************************
  if( logdmp ){  std::cout << "\nBunch Part." << std::endl;}
  // ************************************************************************

  ba.setG(G);         // proton G factor
  
  if( logdmp ){  cout << "gamma = " << gamma << ",  Ggamma = " << G*gamma << endl;}
 
  // int Npart = 128;
  int Npart = 8; 

  if(calcPhaseSpace == 3){
  std::ifstream distInput("./data2/dist2.in");
  distInput >> Npart;
  for(int i=0; i<=Npart; i++){
    distInput >> X[i]>> PX[i] >> Y[i] >> PY[i] >> CT[i] >> DP[i] >> SX[i] >> SY[i] >> SZ[i];

 std::cout << "from dist2.in rank = " << myrank << " bunch number = " << i << " x0 = " << X[i] << ",  x0p = " << PX[i] << "y0 = " << Y[i] << ",  y0p = " << PY[i] << "ct0 = " << CT[i] << ",  dpp0 = " << DP[i] << "\n";









}
  }

  int npart = Npart/numprocs;
  std::cout << "npart = " << npart << "on rank =" << myrank << "\n";

  PAC::Bunch bunch(npart);               // bunch with one particle
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

  optics.calculate();

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
    std::cout << "dpp0 = " << dpp0 << "\n";
  }

    emit_x = emit_x*UAL::pi*1e-6;
    emit_y = emit_y*UAL::pi*1e-6;
std::cout << "emit_x = " << emit_x << " \n";

  if( calcPhaseSpace){
 
    if( logdmp ){ std::cout << "\nTranverse phase space calculated from emittance" << endl;}
    
   
    //    x0 = sqrt(emit_x*beta_x/(6*gamma))*0.001 + tws.d(0)*dpp0;
    //   x0p = tws.dp(0)*dpp0;
    //  y0 = sqrt(emit_y*beta_y/(6*gamma))*0.001 + tws.d(1)*dpp0;
    // y0p = tws.dp(1)*dpp0;
  
    


  std::cout << "dp(0) = "<<tws.dp(0) << "dp(1) =" << tws.dp(1) << "\n";

  } else {

    if( logdmp ){ std::cout << "\nTranverse phase space directly input" << endl;}

    //   x0 = x00 + tws.d(0)*dpp0;
    // x0p = x00p + tws.dp(0)*dpp0;
    // y0 = y00 + tws.d(1)*dpp0;
    //y0p = y00p + tws.dp(1)*dpp0;
      
    x0 = x00; x0p = x00p; y0 = y00; y0p = y00p;
    ct0 = ct0 + offset*2;
  }

  if( logdmp ){
    std::cout << "\nInitial phase space (including dispersion)" << std::endl; //AUL:17MAR10
    //AUL:17MAR10
  }
   
  int ipp = 0;
  std::cout << "npart =" << npart << "\n";
  // index for phases, weights, and dp/p
  int ipsi,iw,idp;
  // number of phase angles psi between y and yprime
  int npsi = 8;
  int nxpsi = 8;
  int ixpsi;
  // number of action weights for gaussian approximation
  int nw = 4;
  // weights for gaussian approximation
  double w[4] = { 0.2671, 0.94, 1.9617, 4.1589};
  double psi_x, psi_y, J_x, J_y, dp0;    
  double dpstep = 0.0003;
  double bunchinY[8] = {0.00993685399984, 0.0100049595104, 0.00421229543076, -0.0040478741835, -0.00993685399984, -0.0100049595104,-0.00421229543076, 0.0040478741835};  
  double bunchinPY[8] = {-0.0411204255547,0.0429006989758, 0.101791175882, 0.101053762486,  0.0411204255547, -0.0429006989758, -0.101791175882, -0.101053762486};



   for(int ip=myrank*npart; ip < npart+myrank*npart; ip++){
    
   // ip = ix + (iy*nx) + iz*(nx*ny)
    
     

     ipsi = ip % npsi;
     ixpsi =((ip-ixpsi)/npsi % nxpsi);
     iw = ((ip-iw)/npsi % nw );
     idp = (ip - ipsi - iw*nw)/(nw*npsi);
     psi_y = (2*UAL::pi/npsi)*ipsi;
     psi_x = (2*UAL::pi/nxpsi)*ixpsi; 
     //J_y = emit_y*exp(-w[iw]/2)*0.5/(6*gamma);
     J_y = emit_y/(6*gamma);
     J_x = emit_x/(6*gamma);

     std::cout << "ip = " << ip << " ipsi = " << ipsi << " iw= " << iw << " idp = " << idp << " iw = " << iw << "w[iw] = " << w[iw] << "Jy = " << J_y << " sqrt() = " << sqrt(J_y*beta_y) << " \n";
     
     x0 = sqrt(J_x*beta_x)*cos(psi_x);
     x0p = sqrt(J_x/beta_x)*(sin(psi_x) + alfa_x*cos(psi_x));

     y0 = sqrt(J_y*beta_y)*cos(psi_y);
     y0p = sqrt(J_y/beta_y)*(sin(psi_y) + alfa_y*cos(psi_y));
     
     dp0 = dpp0 + idp*dpstep;     

     // x0=0.0;x0p=0.0; dp0=0.0;
   bunch[ipp].getPosition().set(x0, x0p, y0, y0p, ct0, dp0);   
    bunch[ipp].setSpin(spin);

    if(calcPhaseSpace == 3){
      std::cout << "setting from dist2.in \n";
    
       std::cout << "from dist2.in rank = " << myrank << " bunch number = " << ip << " x0 = " << X[ip] << ",  x0p = " << PX[ip] << "y0 = " << Y[ip] << ",  y0p = " << PY[ip] << "ct0 = " << CT[ip] << ",  dpp0 = " << DP[ip] << "\n";

      bunch[ipp].getPosition().set(X[ip], PX[ip], Y[ip], PY[ip], CT[ip], DP[ip]);
     spin.setSX(SX[ip]);
     spin.setSY(SY[ip]);
     spin.setSZ(SZ[ip]);  
    
   
    
    bunch[ipp].setSpin(spin);
    }else {
   bunch[ipp].getPosition().set(x0, x0p, y0, y0p, ct0, dp0); 
   
    bunch[ipp].setSpin(spin);
    }
    std::cout << "rank = " << myrank << " bunch number = " << ip << " x0 = " << bunch[ipp].getPosition().getX() << ",  x0p = " << bunch[ipp].getPosition().getPX() << "y0 = " << bunch[ipp].getPosition().getY() << ",  y0p = " <<bunch[ipp].getPosition().getPY() << "ct0 = " << bunch[ipp].getPosition().getCT() << ",  dpp0 = " << bunch[ipp].getPosition().getDE() << "\n";
    ipp++;
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

  // ************************************************************************
  if( logdmp ){  std::cout << "\nTracking. " << std::endl;}
  // ************************************************************************

  double t; // time variable
  
  //  if( logdmp ){ std::cout << "\nTurns = " << turns << std::endl ;}
  std::cout << "\nTurns = " << turns << std::endl ;
  //return 0;

  std::string orbitFile = "./EY20EX800_out/";
  orbitFile += variantName;
  orbitFile += Crank;
  orbitFile += ".orbit";

  PositionPrinter positionPrinter;
  positionPrinter.open(orbitFile.c_str());

  std::string spinFile = "./EY20EX800_out/";
  spinFile += variantName;
  spinFile += Crank;
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
   ipp = 0;
  

   if(iturn % 100 == 0){
  for(int ip=myrank*npart; ip < npart+myrank*npart; ip++){

    if(myrank == 0)std::cout << iturn <<  " "  ;
       positionPrinter.write(iturn, ipp, bunch);
       spinPrinter.write(iturn, ipp, bunch);
   
       ipp++;
  }
   }
    ap -> propagate(bunch);
    
   

  }

  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;

  positionPrinter.close();
  spinPrinter.close();

  MPI_Finalize();
  return 1;
}

