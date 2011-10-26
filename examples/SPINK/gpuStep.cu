#include <iostream>
#include <fstream>
#include <iomanip>
#include <gsl/gsl_randist.h>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"
//#include "../../codes/common/book.h"
#include "SPINK/Propagator/DipoleTracker.hh"
#include "SPINK/Propagator/GpuTracker_hh.cu"
//#include "Def.cu"
//#include "TEAPOT/Integrator/RFCavityTracker.hh"
//#include "SPINK/Propagator/RFCavityTracker.hh"
#include "SPINK/Propagator/SpinTrackerWriter.hh"
//#include "SPINK/Propagator/SnakeTransform.hh"

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
 
  std::ifstream configInput("./datagpu/spinkStep.in");//AULNLD:07JAN10

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
  double emit_z; // Pi 
  double sigct, sigx,sigy,sigxp,sigyp,sigt;
  double x00,x00p,y00,y00p,dpp0,ct0; 
  int calcPhaseSpace;
  bool snkflag ; //AUL:10MAR10
  double mu1; double mu2; double phi1; double phi2; double the1; double the2;
  double stepsize;
  int turns, NPart, dumpstep;

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
  configInput >> dummy >> emit_x >> emit_y >> sigt;
  configInput >> dummy >> x00 >> x00p >> y00 >> y00p >> ct0 >> dpp0;
  configInput >> dummy >> calcPhaseSpace; 
  configInput >> dummy >> snkflag; 
  configInput >> dummy >> mu1 >> mu2 ; 
  configInput >> dummy >> phi1 >> phi2 ; 
  configInput >> dummy >> the1 >> the2 ;
  configInput >> dummy >> turns >> dumpstep;
  configInput >> dummy >> NPart;
  configInput >> dummy >> stepsize;

  std::cout << "emit =" << emit_x << " " << emit_y << " " << sigt << "\n";
  std::cout << "Dist =" << calcPhaseSpace << " snkflag = " << snkflag << " \n";
  std::cout << "mu = " << mu1 << " " << mu2 << " theta = " << the1 << " " << the2 << " turns = " << turns << " dumpstep =" << dumpstep << " NPart =" << NPart << " \n";
  std::cout << "stepsize = " << stepsize << " \n"; 
 // SPINK::SnakeTransform::setOutputDump(outdmp); //AUL:01MAR10
  SPINK::GpuTracker::setOutputDump(outdmp); //AUL:02MAR10
  // SPINK::RFCavityTracker::setOutputDump(outdmp); //AUL:27APR10
  // ************************************************************************
  if( logdmp ){std::cout << "\nDefine the space of Taylor maps." << std::endl;}
  // ************************************************************************

  shell.setMapAttributes(Args() << Arg("order", 5));

  // ************************************************************************
  if( logdmp ){  std::cout << "\nBuild lattice." << std::endl;}
  // ************************************************************************

  std::string sxfFile = "./datagpu/";
  sxfFile += variantName;
  sxfFile += ".sxf";

  std::cout << "sxfFile = " << sxfFile << endl;

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

  std::string outputFile = "./outgpu/cpp/";
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

  std::string mapFile = "./outgpu/cpp/";
  mapFile += variantName;
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
  
  std::string twissFile = "./outgpu/cpp/";
  twissFile += variantName;
  twissFile += ".twiss";

  if( logdmp ){  std::cout << " twiss " << std::endl;}

  std::cout << "we are here \n";
  //shell.twiss(Args() << Arg("print", twissFile.c_str()));

  std::cout << "we are here next \n";
  std::cout << " calculate suml" << std::endl;
  shell.analysis(Args());

  SPINK::GpuTracker::setStep(stepsize);
  // ************************************************************************
  std::cout << "\nAlgorithm Part. " << std::endl;
  // ************************************************************************

  std::string apdfFile = "./datagpu/spink_gpu.apdf";

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
    cout << "Volt = " << V << ", harmon =" << harmon << ", lag = " << lag << std::endl;
    double gamt = 24.5; // RHIC Transition gamma;
    double betak = sqrt(1.0 - 1.0/(gamma*gamma));
    double alpham = 1/(gamt*gamt); 
    double eta = alpham - 1.0/(gamma*gamma);
    double Qs0 = harmon*V*fabs(eta*cos(lag));
    Qs0 /= 2.0*UAL::pi*sqrt(betak)*energy;
    Qs0 = sqrt(Qs0);
    std::cout << "Qs0 = " << Qs0 <<  "\n";
    double sigdp = Qs0*(2.0*UAL::pi/T_0)*sigt/eta;
    std::cout << "sigdp = " << sigdp << "\n";
    std::cout << "eta =" << eta << " \n";
    std::cout << "sigt =" << sigt << "\n";
    emit_z = sigt*sigdp*UAL::pi;
    std::cout << "emit_z = " << emit_z << " \n";  
  

  SPINK::GpuTracker::setRF(V,harmon,lag);
  //  TEAPOT::RFCavityTracker  tracker;
  //tracker.setRF(V, harmon, lag);  //AUL:17MAR10
  //double circ = circum;
   SPINK::GpuTracker::setCircum(circum); //AUL:17MAR10



  // ************************************************************************
  if( logdmp ){  std::cout << "\nBunch Part." << std::endl;}
  // ************************************************************************

  ba.setG(G);         // proton G factor
  
  if( logdmp ){  cout << "gamma = " << gamma << ",  Ggamma = " << G*gamma << endl;}

 
  PAC::Bunch bunch(NPart);               // bunch with one particle
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

 
    std::cout << "beta_x = " << beta_x << "  beta_y = " << beta_y << std::endl;
    std::cout << "alfa_x = " << alfa_x << "  alfa_y = " << alfa_y << std::endl;
    std::cout << "Q_x = " << q_x << "  Q_y = " << q_y << std::endl;
    std::cout << "chrom_x = " << chrm_x << "  chrom_y = " << chrm_y << std::endl;
   
  
    if(calcPhaseSpace >= 1){


      emit_x = emit_x*UAL::pi*1e-6; // /(gamma*6.0);
      emit_y = emit_y*UAL::pi*1e-6; // /(gamma*6.0);
    std::cout << "emit_x = " << emit_x << " emit_y = " << emit_y << " \n";
    
    
    double dp0 = 0.0;
    std::cout << "dp(0) = "<<tws.dp(0) << "dp(1) =" << tws.dp(1) << "\n";
    // ct0 = ct0 + offset*2;
    dp0 = sigdp;
    
 double gama_x = (1.0 + alfa_x*alfa_x)/beta_x;
 double gama_y = (1.0 + alfa_y*alfa_y)/beta_y;
 sigx  = sqrt(beta_x*emit_x/gamma + (tws.d(0)*sigdp)* (tws.d(0)*sigdp));
 sigxp = sigx*sqrt(gama_x/beta_x);
 sigy  = sqrt(beta_y*emit_y/gamma + (tws.d(1)*sigdp)*(tws.d(1)*sigdp));
 sigyp = sigy*sqrt(gama_y/beta_y);

 double corr_x = -alfa_x/sqrt(beta_x*gama_x);
 double corr_y = -alfa_y/sqrt(beta_x*gama_y);

 std::cout << "sigmas = " << sigx << " " << sigxp << " "<< sigy << " " << sigyp << " \n";
 std::cout << "corr_x = " << corr_x << "corr_y = " << corr_y << " \n";


 
 
  // index for phases, weights, and dp/p
 
  // number of phase angles psi between y and yprime
  int Npx,Npy,Npz,Nwx,Nwy,Nwz;
  Npx=Npy=8; Npz=0;
  // number of action weights for gaussian approximation
  int nw = 4;
  Nwx=Nwy=nw; Nwz=0;
  // weights for gaussian approximation
  double w[4] = { 0.2671, 0.94, 1.9617, 4.1589};
  double psi_x, psi_y, psi_z, J_x, J_y, J_z;    
 
  if(calcPhaseSpace == 1){

    std::cout << "Doing Random gaussian distribution \n";

  const gsl_rng_type *Tx, *Ty, *Ts;
  gsl_rng            *rx, *ry, *rs;
  double     rngx,rngxp,rngy,rngyp,rngs,rngdp;
  double bdry = 2.0;
  /* create a generator chosen by the environment variable GSL_RNG_TYPE */
  gsl_rng_env_setup();
  /* gsl_rng_default_seed = (long)getpid(); */
  gsl_rng_default_seed = 103;  Tx = gsl_rng_default;  rx = gsl_rng_alloc(Tx);
  //gsl_rng_set(rx, param->irandx);
  gsl_rng_default_seed = 10555;  Ty = gsl_rng_default;  ry = gsl_rng_alloc(Ty);
  //gsl_rng_set(ry, param->irandy);
  gsl_rng_default_seed = 1178;  Ts = gsl_rng_default;  rs = gsl_rng_alloc(Ts);
  
  for(int ip=0; ip < bunch.size(); ip ++ ){
 

  
    
do {
      gsl_ran_bivariate_gaussian(rx , 1.0, 1.0, corr_x, &rngx, &rngxp);
 } while ((rngx*rngx+rngxp*rngxp) > bdry);
    do { 
      gsl_ran_bivariate_gaussian(ry , 1.0, 1.0, corr_y, &rngy, &rngyp);
    } while( (rngy*rngy+rngyp*rngyp) > bdry);
    do {
      gsl_ran_bivariate_gaussian(rs , 1.0, 1.0, 0.0, &rngs , &rngdp);
    } while ( (rngs*rngs+rngdp*rngdp) > bdry);
    
  sigct = sigt*cc;
    x0  = rngx *sigx;
    x0p = rngxp*sigxp;
    y0  = rngy *sigy;
    y0p = rngyp*sigyp;
    ct0  = rngs *sigct + offset*2;
    dp0 = rngdp*sigdp;


 bunch[ip].getPosition().set(x0, x0p, y0, y0p, ct0, dp0);    //AUL:17MAR10
    bunch[ip].setSpin(spin);
     
    
      std::cout << ip << "  ";
      std::cout << " " << x0 << ",  " << x0p ;
     std::cout << " " << y0 << ", " << y0p ;
     std::cout << " " << ct0 << ", " << dp0 << std::endl;
   
  }
  
 // if(restart){
   // fclose (pFile);}
    gsl_rng_free (rx);
  gsl_rng_free (ry);
  gsl_rng_free (rs);} else { // Do discrete Gaussian distribution

    std::cout << "Doing Discrete Gaussian Distribution \n";

int ip = 0;

 std::cout << "Nwx = " << Nwx << " Nwy = " << Nwy << " Nwz =" << Nwz << " \n";
 std::cout << "Npx = " << Npx << " Npy = " << Npy << " Npz = " << Npz << " \n";

for(int iwx=0; iwx < Nwx; iwx ++){
    for(int iwy=0; iwy < Nwy; iwy ++){
       for(int ipx=0; ipx < Npx; ipx ++){
          for(int ipy=0; ipy < Npy; ipy++){
	    // for(int ipz=0; ipz < Npz; ipz++){
            //  for(int iwz=0;iwz < Nwz; iwz ++){
	    




   int	ipz = 0; int iwz = 0;
     psi_x = (2*UAL::pi/Npx)*ipx;
     psi_x = (2*UAL::pi/Npy)*ipy;
     psi_z = (2*UAL::pi/Npz)*ipz; 
     J_y = emit_y*exp(-w[iwy]/2)*0.5/(gamma);
     J_x = emit_x*exp(-w[iwx]/2)*0.5/(gamma);
     J_z = emit_z*exp(-w[iwz]/2)*0.5;

     x0 = sqrt(J_x*beta_x)*cos(psi_x);
     x0p = sqrt(J_x/beta_x)*(sin(psi_x) + alfa_x*cos(psi_x));
    

     y0 = sqrt(J_y*beta_y)*cos(psi_y);
     y0p = sqrt(J_y/beta_y)*(sin(psi_y) + alfa_y*cos(psi_y));
     
     ct0 =  offset*2;
     dp0 = 0.0;     

   bunch[ip].getPosition().set(x0, x0p, y0, y0p, ct0, dp0);    //AUL:17MAR10
    bunch[ip].setSpin(spin);
     
    
      std::cout << ip << "  ";
      std::cout << " " << x0 << ",  " << x0p ;
     std::cout << " " << y0 << ", " << y0p ;
     std::cout << " " << ct0 << ", " << dp0 << std::endl;
        ip++;
	    }}}}
//}}   
  
  }   

   // end of calcPhaseSpace 
    }else {
      std::cout << " starting loading of dist.in \n";
      precision Ggam,sx0,sy0,sz0,dp0;
      // else read in particle distribution from file
      std::ifstream distInput("dist.in");
      for(int ip=0; ip< bunch.size(); ip++){
	distInput >> gamma >> Ggam >> sx0 >> sy0 >> sz0 >> x0 >> x0p >> y0 >> y0p >> ct0 >> dp0;
	 std::cout << ip << "  ";
      std::cout << " " << x0 << ",  " << x0p ;
     std::cout << " " << y0 << ", " << y0p ;
     std::cout << " " << ct0 << ", " << dp0  <<  " ," << sx0 << ", " << sy0 << " ," << sz0 << std::endl;

       bunch[ip].getPosition().set(x0, x0p, y0, y0p, ct0, dp0);    //AUL:17MAR10
       //  bunch[ip].setSpin(spin);
    //  bunch[ip].getPosition().set(x0,x0p,y0,y0p,ct0,dp0);
	       spin.setSX(sx0);
		spin.setSY(sy0);
		spin.setSZ(sz0);
    	bunch[ip].setSpin(spin);
      }
      energy = gamma*mass;
  std::cout << "setting energy from file \n";
  
   shell.setBeamAttributes(Args() << Arg("energy", energy) << Arg("mass", mass)
  			  << Arg("charge",charge));
    std::cout << "after setting energy from file \n";
   ba = shell.getBeamAttributes();
   bunch.setBeamAttributes(ba);   

    }

  /** read in snake parameters AULNLD 2/9/10 */
 
 //SPINK::GpuDipoleTracker::loadPart(bunch);
 
  if( snkflag ){  //AUL:07MAY10

    SPINK::GpuTracker::setSnakeParams(mu1, mu2, phi1, phi2, the1, the2);

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

  std::string orbitFile = "./outgpu/cpp/";
  orbitFile += variantName;
  orbitFile += ".orbit";

  PositionPrinter positionPrinter;
  positionPrinter.open(orbitFile.c_str());

  std::string spinFile = "./outgpu/cpp/";
  spinFile += variantName;
  spinFile += ".spin";
  
  SpinPrinter spinPrinter;
  spinPrinter.open(spinFile.c_str());
  int count;
  int step = 1;
  ba.setElapsedTime(0.0);

  start_ms();

  std::ofstream allpart, avgpart;

  //  allpart.open("PartOut.dat");
  avgpart.open("AvgOut.dat");
  char line[200];
  int N = bunch.size();
  precision Ggam = G*gamma;
  precision SxAvg =0.00, SyAvg=0.00, SzAvg=0.00;
  int stepdump = 0;
  
  SPINK::GpuTracker::setNturns(step);

 





 
   for(int iturn = 1; iturn <= turns; iturn++){

    /** to pass turn no for diagnostics AUL:02MAR10 */
    //  SPINK::SnakeTransform::setNturns(iturn);
  
    // SPINK::RFCavityTracker::setNturns(iturn);//AUL:27APR10
// for(int ip=0; ip < bunch.size(); ip++){
  //     positionPrinter.write(iturn, ip, bunch);
    //   spinPrinter.write(iturn, ip, bunch);
   // }

 
    
   SPINK::GpuTracker::GpuProp(bunch);

   // ap->propagate(bunch);
   //if( iturn % 10 == 0 ){
    avgpart << iturn*step << " ";
    // SPINK::GpuTracker::GpuPropagate(bunch);
    SPINK::GpuTracker::readPart(bunch,0);
     gamma = Energy[0]/mass;
     Ggam  = gamma*G; 
     SxAvg = 0.00; SyAvg=0.00; SzAvg=0.00;
     count = 0;
     // if(iturn == 5) { std::cout << "setting new step size to" << stepsize << "\n";
       
       // }

     //  if(iturn == 10) { SPINK::GpuTracker::setStep(stepsize*20);}
    //} 
         for(int ip=0; ip < bunch.size(); ip++){
         
   if(pos[ip].x*pos[ip].px*pos[ip].y*pos[ip].py*pos[ip].ct*pos[ip].de != pos[ip].x*pos[ip].px*pos[ip].y*pos[ip].py*pos[ip].ct*pos[ip].de ){ 
     }else {count++;
     SxAvg += pos[ip].sx; SyAvg += pos[ip].sy; SzAvg += pos[ip].sz;

     }
       }
	 int ip = 0;
	 sprintf(line," %i  %e  %e  %e  %e  %e  %e  %e  %e  %e  %e  %e \n",count,gamma,Ggam,SxAvg/count,SyAvg/count,SzAvg/count,pos[ip].x,pos[ip].px,pos[ip].y,pos[ip].py,pos[ip].ct,pos[ip].de);
	 avgpart << line ;


	 if(iturn % dumpstep == 0) {
           std::string PartFile = "PartOut";
           char dumpNo[3];
	   sprintf(dumpNo, "%d",stepdump);
	   stepdump++;
           PartFile += dumpNo;
           PartFile += ".dat";
           const char * inputc = PartFile.c_str();

           allpart.open(inputc);
for(int ip = 0; ip < N; ip++) {
    	 sprintf(line," %e  %e  %e  %e  %e  %e  %e  %e  %e  %e  %e \n",gamma,Ggam,pos[ip].sx,pos[ip].sy,pos[ip].sz,pos[ip].x,pos[ip].px,pos[ip].y,pos[ip].py,pos[ip].ct,pos[ip].de);
	 allpart << line ;
     }

   allpart.close();


	 }




  // }

    }
   // SPINK::GpuTracker::readPart(bunch,1);



     allpart.open("PartOut_last.dat");

   for(int ip = 0; ip < N; ip++) {
    	 sprintf(line," %e  %e  %e  %e  %e  %e  %e  %e  %e  %e  %e \n",gamma,Ggam,pos[ip].sx,pos[ip].sy,pos[ip].sz,pos[ip].x,pos[ip].px,pos[ip].y,pos[ip].py,pos[ip].ct,pos[ip].de);
	 allpart << line ;
     }

   allpart.close();
   avgpart.close();
  t = (end_ms());
  std::cout << "time  = " << t << " ms" << endl;
  positionPrinter.close();
  spinPrinter.close();
  //  cudaThreadExit();

  return 1;
}


