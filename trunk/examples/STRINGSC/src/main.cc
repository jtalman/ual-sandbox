#include <fstream>
#include <vector>

#include "math.h"
#include "timer.h"

#include "Survey/PacSurveyData.h"

#include "UAL/UI/Shell.hh"
#include "PAC/Beam/Bunch.hh"
#include "Optics/PacTwissData.h"

#include "Optics/PacTMap.h"

#include "Main/Teapot.h"
#include "Integrator/TeapotElement.h"

#include "ACCSIM/Bunch/BunchGenerator.hh"
#include "ACCSIM/Bunch/BunchAnalyzer.hh"

#include "UAL/UI/Shell.hh"
#include "UAL/UI/OpticsCalculator.hh"

#include "TEAPOT/StringSC/StringSCSolver.hh"

std::ofstream forcecomp("../out/forcecomp");
std::ofstream HTangles("../out/HTangles");
std::ofstream moments("../out/moments");

using namespace UAL;

int main(int argc, char** argv) {

  // Input parameters

  // ************************************************************************
  std::cout << "\nEcho input parameters." << std::endl;
  // ************************************************************************

  if(argc < 3) {
    std::cout << "Usage : ./main <machine> <qtot> <np> <nturns> " 
	      << "<seed> <ee> <xhW> <yhW> <cthW> <dehW> <lstr> <longit_dist> <ddeBydct> <d2deBydct2> <betax0> <alphax0> ,betay0> <alphay0>" << std::endl;
    std::cout << "All units are M.K.S. except ee which is in GeV" << std::endl;

    exit(1);
  }

  // Command lines for debugging
  // ./main CTFII-2 1.0e-26 2 1 -100 0.04 0.0012 0.0023 0.0012 0.0001 0.0001 test -18.0 0 1.687 -2.65 5.808 -6.389
  // result in ../out/bunchout
  // -------------------------
  // -1.36993e-09 9.21274e-10 -0.000144372 -0.000209667 1.07393e-05 0.04004
  // 2.7733e-18 -4.29622e-19 -2.79313e-19 -1.49442e-19 2.66917e-15 0.04

  // ../linux/scripts/testm_tiny
  // result in ...out/xmgrace/bunch/outputs/gaussian/chicane.data
  // ------------------------------------------------------------
  // 100
  // chicaneR56q25   -1.086  34.47   1.765   1.005   0.01887 1.845   1.026

  // sxf file 
  std::string machine  = argv[1];
  // std::string sxfInputFile   = "../data/CTFII-4.sxf"; // "../data/" + machine + ".sxf";
  std::string sxfInputFile   = "../data/" + machine + ".sxf";

  // total bunch charge (Coulombs)
  double qtot = atof(argv[2]); 

  // number of macroparticles.
  int Np = atoi(argv[3]);
  int Nph = Np/2;

  if( Np != Nph + Nph ){
    std::cout << "Np must be even" << std::endl;
    exit(1);
  }

  // number of turns 
  int Nturns = atoi(argv[4]);
            
  // seed 
  int seed = atoi(argv[5]);
  int seedSave = seed;

  // electron energy
  double ee = atof(argv[6]);

  // initial horizontal half width
  double xhW = atof(argv[7]);

  // initial vertical half width
  double yhW = atof(argv[8]);

  // initial half bunch length
  double cthW = atof(argv[9]);

  // initial half fractional energy offset
  double dehW = atof(argv[10]);

  // half string length
  double lstr = atof(argv[11]);

  // longitudinal distribution (uniform, uniform-gridded, gaussian, gaussian-gridded, or test)
  // for longit_dist="test" the distribution is uniform and not symmetrized
  std::string longit_dist  = argv[12];

  // initial  de/d(ct)  correlation 
  double ddeBydct = atof(argv[13]);

  // initial  d2e/d(ct)^2ct  correlation
  double d2deBydct2 = atof(argv[14]);

  // R56q is adjusted to match Braun, Corsini et al. PRST-AB 3, 124402 (2000),
  //         R56q = k_CTFII * theta^2, where k_CTFII = 361 mm/rad^2
  // NOT from UAL lattice calculation
  
  // double R56q = atof(argv[15]);
  // double bendfac = sqrt(R56q/8.03);

  double betax0 = atof(argv[15]);
  double alphax0 = atof(argv[16]);
  double betay0 = atof(argv[17]);
  double alphay0 = atof(argv[18]);

  std::cout << " qtot="   << qtot << " C"
	    << ", SYMMETRIZED Np="     << Np 
	    << ", Nturns=" << Nturns 
	    << ", seed = " << seedSave
	    << "\n ee = " << ee
	    << "\n xhW = " << xhW
	    << ", yhW = " << yhW
	    << ", cthW = " << cthW
	    << ", dehW = " << dehW 
	    << ", ddeBydct = " << ddeBydct
            << "\n " << longit_dist << " longitudinal"
	    << ", lstr = " << lstr 
	    << ", betax0 = " << betax0 
	    << ", alphax0 = " << alphax0 
	    << ", betay0 = " << betay0 
	    << ", alphay0 = " << alphay0 
            << std::endl; 

  // moments << "\n\n" << "# R56q = " << std::endl;

  // ************************************************************************
  std::cout << "\nDefine the space of Taylor maps." << std::endl;
  // ************************************************************************

  UAL::Shell shell;

  shell.setMapAttributes(Args() << Arg("order", 5)); 

  // ************************************************************************
  std::cout << "\nRead SXF file (lattice description) \"" << sxfInputFile << "\"" << std::endl;
  // ************************************************************************

  // std::string sxfEchoFile = "./" + machine + ".sxf";
  std::string sxfEchoFile = "./echo.sxf";
  
    bool status = shell.readSXF(Args() 
			 << Arg("file",  sxfInputFile) 
			 << Arg("print", sxfEchoFile));
  if(!status) exit(1);

  PacLattices::iterator latIterator = PacLattices::instance()->find("ring");
  PacLattice& lattice = *latIterator;

  for(int i=0; i < lattice.size(); i++){
    if(lattice[i].type() == "Quadrupole"){
      lattice[i].addN(2.0);
    }
    //    if(lattice[i].type() == "Sbend"){
    //  lattice[i].addN(1.0);
    // }
  }

  // ************************************************************************
  std::cout << "\nSelect lattice." << std::endl;
  // ************************************************************************

  status = shell.use(Args() << Arg("lattice", "ring"));
  if(!status) exit(1);

  // ************************************************************************
  std::cout << "\nSet beam attributes." << std::endl;
  // ************************************************************************

  // RT, 25 June, 2004. On input "qtot" is expected to be the total bunch
  // charge in Coulombs. The number of macroparticles is "Np"
  // So "macrosize = qtot/Np" is the charge (in Coulombs) of one macroparticle.
  // At least this is what 
  //     setBeamAttributes( ... << Arg("macrosize", qtot/Np));
  // provides.
  // In "StringSCSolver.cc" the line
  //     m_macrosize = bunch.getBeamAttributes().getMacrosize();
  // presumably recovers "qtot/Np"
  // It is still not clear to me whether or not this usage is consistent with
  // the usage of "macrosize" elsewhere in UAL.

  shell.setBeamAttributes (Args() 
			   << Arg("mass", UAL::emass) 
			   << Arg("energy", ee)          // ee=5.11 gives gamma=10000
			   << Arg("macrosize", qtot/Np));

  PAC::BeamAttributes ba = shell.getBeamAttributes();

  // ************************************************************************
  std::cout << "\nRead APDF file (propagator description). " << std::endl;
  // ************************************************************************

  status = shell.readAPDF(Args() << Arg("file", "../data/stringsc.apdf"));
  if(!status) exit(1);

  // ************************************************************************
  std::cout << "\nDo survey. " << std::endl;
  // ************************************************************************

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  PacSurveyData surveyData;
  optics.m_teapot->survey(surveyData);

  double suml = surveyData.survey().suml();

  std::cout << "suml = " << suml << std::endl;

  // ************************************************************************
  std::cout << "\nAssign input twiss values. " << std::endl;
  // ************************************************************************

  std::ofstream out("twiss.out");

  std::vector<std::string> columns(11);
  columns[0]  = "#";
  columns[1]  = "suml";
  columns[2]  = "betax";
  columns[3]  = "alfax";
  columns[4]  = "mux";
  columns[5]  = "dx";
  columns[6]  = "betay";
  columns[7]  = "alfay";
  columns[8]  = "muy";
  columns[9] = "dy";
  columns[10]  = "name";

  char endLine = '\0';

  double twopi = 2.0*UAL::pi;

  out << "#-----------------------------------------------------------";
  out << "------------------------------------------------------------" << std::endl; 

  char line[200];
  sprintf(line, "%-5s %-10s   %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s%c", 
	columns[0].c_str(),  columns[1].c_str(), columns[2].c_str(), columns[3].c_str(),  
	columns[4].c_str(),
	columns[5].c_str(), columns[6].c_str(), columns[7].c_str(), columns[8].c_str(),  
	columns[9].c_str(), columns[10].c_str(), endLine);
  out << line << std::endl;

  out << "#-----------------------------------------------------------";
  out << "#-----------------------------------------------------------" << std::endl; 

  //  OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();
  Teapot* teapot = optics.m_teapot;

  PAC::Position orbit;
  // teapot->clorbit(orbit, ba);

  PacTMap map(6);
  map.refOrbit(orbit);
  teapot->map(map, ba, 1);

  PacTwissData twiss;
  double mux = 0.0, muy = 0.0;
  twiss.mu(0, mux);
  twiss.beta(0, betax0);
  twiss.alpha(0, alphax0);
  twiss.mu(1, muy);
  twiss.beta(1, betay0);
  twiss.alpha(1, alphay0);

  double betax  = twiss.beta(0);
  double betay  = twiss.beta(1);
  double alphax = twiss.alpha(0);
  double alphay = twiss.alpha(1);

  double betaxin = betax;
  double betayin = betay;
  double alphaxin = alphax;
  double alphayin = alphay;

  std::cout << "  betaxin = " << betaxin
	    << ", betayin = " << betayin
	    << std::endl
	    << "  alphaxin = " << alphaxin 
	    << ", alphayin = " << alphayin
	    << std::endl;

  // ************************************************************************
  std::cout << "\nPrepare a bunch of " << Np << " particles. " << std::endl;
  std::cout << "(The second half will be overwritten later) " << std::endl;
  // ************************************************************************ 

  PAC::Bunch bunch(Np);

  double e = ba.getEnergy();
  double m = ba.getMass();
  double gam = e/m;
  double v0byc = sqrt(e*e - m*m)/e;
  ba.setRevfreq(v0byc*UAL::clight/suml);

  double avX=0;
  double avPX=0;
  double avY=0;
  double avPY=0;
  double avCT=0;
  double avDE=0;

  double p_epsxin;
  double p_epsyin;
  double epsxin;
  double epsyin;
  double avDEin;
  double rmsdein;
  double rmsctin;

  bunch.setBeamAttributes(ba);

  // set bunch distribution

  ACCSIM::BunchGenerator bunchGenerator;
  ACCSIM::BunchAnalyzer  bunchAnalyzer;

  //  double xhW = 0.0002;
  double ex  = xhW*xhW/betaxin;

  //  double yhW = 0.0002;
  double ey  = yhW*yhW/betayin;

  //  double cthW = 0.0005;
  //  double dehW = 0.001;

  if( Np == 1 ){
    double pxhW = xhW/betaxin;
    double pyhW = yhW/betayin;
    bunch[0].getPosition().set(xhW, pxhW, yhW, pyhW, cthW, dehW);
  }

  else if( Np == 2 ){
    // Case Np==2 is handled differently than other Np values
    // for convenience in checking the stringsc formulas
    // test
    bunch[1].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    // source
    bunch[0].getPosition().set(0.0, 0.0, 1.0e-4, 0.0, 0.0, 1.e-3);
  }

  else{

    // Transverse distribution

    std::cout << "\nThe quantities returned by 'bunchAnalyzer.getRMS' " 
	      << "and labeled 'rmsx' and 'rmsy'\n in 'main.cc' are not, " 
	      << "in fact, rms., values.\nWhat is returned are 'emittance_x'" 
	      << " and 'emittance_y'. \nBut 'rmsct' and 'rmsde' are truly r.m.s. values";

    std::cout << "\nThe 'p_' in 'p_epsx' and 'p_epsy' stands for 'pseudo'.\n"
              << std::endl;

    double mFactor = 3;
    std::cout << "desired: p_epsx = " << ex << ", p_epsy = " << ey << std::endl;

    // Total emittance (the second and fourth elements are not used). 
    PAC::Position emittance;
    emittance.set(ex*8, 0.0, ey*8, 0.0, 0.0, 0.0);

    bunchGenerator.addBinomialEllipses(bunch, mFactor, twiss, emittance, seed);

    // Longitudinal distribution
    // Default: ACCSIM idistl = 4 : uniform in phase

    PAC::Position halfWidth;

    if( (longit_dist=="uniform") || (longit_dist=="test") ){
      halfWidth.set(0.0, 0.0, 0.0, 0.0, cthW, dehW);
      bunchGenerator.addUniformRectangles(bunch, halfWidth, seed);
    }
    else if(longit_dist=="uniform-gridded"){
      halfWidth.set(0.0, 0.0, 0.0, 0.0, 0.0, dehW);
      bunchGenerator.addUniformRectangles(bunch, halfWidth, seed);
      {
	int j;
	double ct;
	double dct = cthW/Nph;
	for(j = 0; j < Np; j++){
	  ct = -cthW + dct*(0.5+j);
	  bunch[j].getPosition().setCT(ct);
	  }
      }
    }
    else if(longit_dist=="gaussian"){
      halfWidth.set(0.0, 0.0, 0.0, 0.0, cthW, dehW);
      PAC::Position sig;
      double CUT=3.0;
      sig.set(0.0, 0.0, 0.0, 0.0, cthW, dehW);
      bunchGenerator.addGaussianRectangles(bunch, sig, CUT, seed);
    }
     else if(longit_dist=="gaussian-gridded"){
      PAC::Position sig;
      double CUT=3.0;
      sig.set(0.0, 0.0, 0.0, 0.0, 0.0, dehW);
      bunchGenerator.addGaussianRectangles(bunch, sig, CUT, seed);
      {
	int j, jrefl;
	double p;
	double dp = 1.0/Nph;
	double pi = 3.14159265359;
	double ct;
        double a = 8.0/3.0/pi*(pi-3.0)/(4.0-pi);

	for(j = 0; j < Nph; j++){
	  jrefl = Np-j-1;

	  p = 1 - dp*(0.5+j);

	  // Approximate the inverse error function using Sergei Winitzki, 
          // "A handy approximation for the error function and its inverse", 2006

	  ct = sqrt(-2.0/pi/a - 0.5*log(1-p*p) + sqrt((2/pi/a + 0.5*log(1-p*p))*(2/pi/a + 0.5*log(1-p*p)) - log(1-p*p)/a));
	  ct *= sqrt(2.0)*cthW;

	  bunch[j].getPosition().setCT(-ct);
	  bunch[jrefl].getPosition().setCT(ct);
	  }
      }
    }
    else {
    std::cout << " Not a legal longitudinal distribution." << std::endl;
    exit(1);
    }

  // ************************************************************************
  std::cout << "\nCopy reflected first half bunch to second half" << std::endl;
  // ************************************************************************ 

  if ( longit_dist != "test" ){
    PAC::Position ph = bunch[0].getPosition();
    for(int iph=0; iph < Nph; iph++){
      int ipr = iph+Nph;

      ph = bunch[iph].getPosition();

      bunch[ipr].getPosition().setX(-ph.getX());
      bunch[ipr].getPosition().setPX(-ph.getPX());
      bunch[ipr].getPosition().setY(-ph.getY());
      bunch[ipr].getPosition().setPY(-ph.getPY());
    }
  }

  // ************************************************************************
  std::cout << "\nCheck emittances of full bunch " << std::endl;
  // ************************************************************************ 

    PAC::Position rms, orbit;
    PacTwissData twissOut;

    bunchAnalyzer.getRMS(bunch, orbit, twissOut, rms);

    p_epsxin = rms.getX();
    p_epsyin = rms.getY();
    rmsctin = rms.getCT();
    rmsdein = rms.getDE();

    std::cout << "achieved: p_epsx = " << p_epsxin
	      << ", p_epsy = " << p_epsyin << std::endl;
    std::cout << "achieved: p_epsx_n = " << p_epsxin*gam
	      << ", p_epsy_n = " << p_epsyin*gam << std::endl;
    std::cout << "achieved: betax = " << twissOut.beta(0) 
	      << ", betay = " << twissOut.beta(1) << std::endl;
    std::cout << "achieved: de_rms = " << rmsdein 
              << ", ct_rms = " << rmsctin << std::endl;

  // ************************************************************************
  std::cout << "\nIntroduce 'de' as function of 'ct'. ddeBydct = " 
            << ddeBydct << ", d2deBydct2 = " << d2deBydct2 << std::endl;
  // ************************************************************************ 

    // Alter the bunch by augmenting the "de" component 
    // by a term proportional to "ct"
    double ctt, det;
    for(int ip=0; ip < Np; ip++) {
    ctt = bunch[ip].getPosition().getCT();
    det = bunch[ip].getPosition().getDE();
    det += ctt*ddeBydct;
    det += 0.5*ctt*ctt*d2deBydct2;
    bunch[ip].getPosition().setDE(det);
    }

    bunchAnalyzer.getRMS(bunch, orbit, twissOut, rms);
    rmsdein = rms.getDE();
    std::cout << "achieved: de_rms = " << rmsdein << std::endl;

  // ************************************************************************
  std::cout << "\nCheck emittance calculation and save 'bunchin'. " << std::endl;
  // ************************************************************************

    std::ofstream bunchin("../out/bunchin");

    avX=0;
    avPX=0;
    avY=0;
    avPY=0;
    avCT=0;
    avDE=0;

    double epsx_acc = 0.0;
    double epsy_acc = 0.0;

    double x;
    double px;
    double y;
    double py;
    double ct;
    double de;

    PAC::Position p = bunch[0].getPosition();
    for(int ip=0; ip < Np; ip++){
      p = bunch[ip].getPosition();
      x = p.getX();
      px = p.getPX();
      y = p.getY();
      py = p.getPY();
      ct = p.getCT();
      de = p.getDE();

      avX += x/Np;
      avPX += px/Np;
      avY += y/Np;
      avPY += py/Np;
      avCT += ct/Np;
      avDE += de/Np;

      epsx_acc += (x*x + (alphaxin*x + betaxin*px)*(alphaxin*x + betaxin*px))/betaxin/2/Np;
      epsy_acc += (y*y + (alphayin*y + betayin*py)*(alphayin*y + betayin*py))/betayin/2/Np;

      bunchin << x << "  " << px << "  " << y << "  " << py << "  " << ct << "  " << e*(1+de) << std::endl;
    }
    epsxin = epsx_acc;
    epsyin = epsy_acc;
    avDEin = avDE;

    std::cout << "input: avX = " << avX
	      << ", avPX = " << avPX << std::endl;
    std::cout << "input: avY = " << avY
	      << ", avPY = " << avPY << std::endl;
    std::cout << "input: avCT = " << avCT
	      << ", avE(MeV) = " << (1+avDE)*e*1000.0 << "\n" << std::endl;

    std::cout << "input: epsxin = " << epsxin  << ", epsyin = " << epsyin << std::endl;
    std::cout << "input: epsxin_n = " << epsxin*gam  << ", epsyin_n = " << epsyin*gam << std::endl;
  }

  // ************************************************************************
  std::cout << "\nDo linear analysis. " << std::endl;
  // ************************************************************************

  double at = 0;
  for(int i = 0; i < teapot->size(); i++){

    PacTMap smap(6);
    smap.refOrbit(orbit);

    teapot->trackMap(smap, ba, i, i + 1);
    teapot->trackTwiss(twiss, smap);

    if((twiss.mu(0) - mux) < 0.0) twiss.mu(0, twiss.mu(0) + 1.0);
    mux = twiss.mu(0);

    if((twiss.mu(1) - muy) < 0.0) twiss.mu(1, twiss.mu(1) + 1.0);
    muy = twiss.mu(1);

    at += teapot->element(i).l();

    sprintf(line, "%5d %15.7e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %-10s%c", 
	    i, at, 
	    twiss.beta(0), twiss.alpha(0), 
	    twiss.mu(0)*twopi, twiss.d(0),
	    twiss.beta(1), twiss.alpha(1), 
	    twiss.mu(1)*twopi, twiss.d(1), 
            teapot->element(i).getDesignName().c_str(), endLine);
    out << line << std::endl;

    if(lattice[i].type() == "Sbend"){
      TeapotElement& te = teapot->element(i);
      std::cout << "sbend " << te.getDesignName() 
		<< ",k1 = " << te.bend()->ke1() 
		<< ",k2 = " << te.bend()->ke2() << std::endl;
    }
  }

  out << "#-----------------------------------------------------------";
  out << "------------------------------------------------------------" << std::endl; 

  sprintf(line, "%-5s  %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s%c", 
	columns[0].c_str(),  columns[1].c_str(), columns[2].c_str(), columns[3].c_str(),  
	columns[4].c_str(),
	columns[5].c_str(), columns[6].c_str(), columns[7].c_str(), columns[8].c_str(),  
	columns[9].c_str(), columns[10].c_str(), endLine);
  out << line << std::endl;

  out << "#-----------------------------------------------------------";
  out << "------------------------------------------------------------" << std::endl; 

  out.close();

  double qx = twiss.mu(0)/(2.*UAL::pi);
  double qy = twiss.mu(1)/(2.*UAL::pi);

  std::cout << "  qx = "   << qx << ", qy = " << qy << std::endl;

  double betaxout = twiss.beta(0);
  double betayout = twiss.beta(1);
  double alphaxout = twiss.alpha(0);
  double alphayout = twiss.alpha(1);

  std::cout << "  betaxout = " << betaxout
	    << ", betayout = " << betayout
	    << std::endl
	    << "  alphaxout = " << alphaxout
	    << ", alphayout = " << alphayout 
	    << std::endl;

  // ************************************************************************
  std::cout << "\nDefine TEAPOT String SC Solver." << std::endl;
  // ************************************************************************

  TEAPOT::StringSCSolver& scSolver =  TEAPOT::StringSCSolver::getInstance();
  scSolver.setStringL(lstr);

  // The following value of "strH" was established using 
  // "Q-1nC-36mu-Nom_yhW", "Q-1nC-36mu-0.1xNom_yhW"
  // to produce horizontal emittance ratio of about 1.4? for R56=25mm. 
  // This required strH=0.01*yhW for yhW=36microns
  // At the same time the arbitrary ten-fold reduction factor was removed from 
  // "StringSCSolver,cc". 
  double strH=36e-8;
  std::cout << "strH: " << strH << "\n" << std::endl;
  scSolver.setStringH(strH); 

  scSolver.setMaxBunchSize(Np);
  // scSolver.setBendfac(bendfac);

  // ************************************************************************
  std::cout << "Track it." << std::endl;
  // ************************************************************************ 

  for(int it=0; it < Nturns; it++){
    // scSolver.setCounter(0);
    shell.run(Args() << Arg("turns", 1) << Arg("bunch", bunch));
  }

  // ************************************************************************
  std::cout << "Print results.\n" << std::endl;
  // ************************************************************************ 

  // R.m.s. spreads are available using ACCSIM method "bunchAnalyzer.getRMS" but
  // it is appropriate to check these calculations independently, especially as
  // regards the bunch length, since the bunch centroid is systematically displaced,
  // both transversely and longitudinally, as a function of the bunch charge.

  std::ofstream bunchout("../out/bunchout");

  avX=0;
  avPX=0;
  avY=0;
  avPY=0;
  avCT=0;
  avDE=0;

  double x;
  double px;
  double y;
  double py;
  double ct;
  double de;

  PAC::Position p2 = bunch[0].getPosition();
  for(int ip=0; ip < Np; ip++){
    p2 = bunch[ip].getPosition();

    x = p2.getX();
    px = p2.getPX();
    y = p2.getY();
    py = p2.getPY();
    ct = p2.getCT();
    de = p2.getDE();

    avX += x/Np;
    avPX += px/Np;
    avY += y/Np;
    avPY += py/Np;
    avCT += ct/Np;
    avDE += de/Np;

    bunchout << x << " " << px << " " << y << " " << py << " " << ct << " " << e*(1+de) << std::endl;
  }

  double epsx_acc = 0.0;
  double epsy_acc = 0.0;

  for(int ip=0; ip < Np; ip++){
    p2 = bunch[ip].getPosition();
    x = p2.getX()-avX;
    px = p2.getPX()-avPX;
    y = p2.getY()-avY;
    py = p2.getPY()-avPY;

    epsy_acc += (y*y + (alphayout*y + betayout*py)*(alphayout*y + betayout*py))/betayout/2/Np;
  }
  double epsyout = epsy_acc;

  /*
  // EPSX_RATIO_MAX = 20*Np/400 barely cuts 19 particles in one test case with Np=800,
  // Cutting one particle shifts epsx fractionally by about -(40*Np/400)/Np=0.1.  
  double EPSX_RATIO_MAX = 40*Np/400;
  int number_cut = 0;
  */

  for(int ip=0; ip < Np; ip++){
    p2 = bunch[ip].getPosition();

    x = p2.getX()-avX;
    px = p2.getPX()-avPX;
    y = p2.getY()-avY;
    py = p2.getPY()-avPY;

    /*
    double epsx_1 = (x*x + (alphaxout*x + betaxout*px)*(alphaxout*x + betaxout*px))/betaxout/2;
    if ( epsx_1/epsxin > EPSX_RATIO_MAX ) {
      x = 0;
      bunch[ip].getPosition().setX(avX);
      px = 0;
      bunch[ip].getPosition().setPX(avPX);
      number_cut++;
    }
    */

    epsx_acc += (x*x + (alphaxout*x + betaxout*px)*(alphaxout*x + betaxout*px))/betaxout/2/Np;
  }
  double epsxout = epsx_acc;

  std::string xmgracestring1("\nwith string\n string on\n string loctype view\n string 0.05, 0.95\n string color 1\n string rot 0\n string font 4\n string just 0\n string char size 0.7\n string def ");
  std::string xmgracestring2("\nwith string\n string on\n string loctype view\n string 0.05, 0.93\n string color 1\n string rot 0\n string font 4\n string just 0\n string char size 0.7\n string def ");
  std::string xmgracestring3("\nwith string\n string on\n string loctype view\n string 0.05, 0.91\n string color 1\n string rot 0\n string font 4\n string just 0\n string char size 0.7\n string def ");
  std::string xmgracestring4("\nwith string\n string on\n string loctype view\n string 0.05, 0.89\n string color 1\n string rot 0\n string font 4\n string just 0\n string char size 0.7\n string def ");

  if( Np > 2 ) {
    PAC::Position rms, orbit;
    PacTwissData twissOut;

    // std::cout << "number_cut = " << number_cut << std::endl;

    bunchAnalyzer.getRMS(bunch, orbit, twissOut, rms);

    double betaxp = twissOut.beta(0);
    double betayp = twissOut.beta(1);

    double p_epsx = rms.getX();
    double p_epsy = rms.getY();
    double rmsde = rms.getDE();
    double rmsct = rms.getCT();

    std::cout << "output: avX = " << avX
	      << ", avPX = " << avPX << std::endl;
    std::cout << "output: avY = " << avY
	      << ", avPY = " << avPY << std::endl;
    std::cout << "output: avCT = " << avCT
	      << ", avE(MeV) = " << (1+avDE)*e*1000.0 << "\n" << std::endl;

    std::cout << "output: epsxout = " << epsxout  << ", epsyout = " << epsyout << std::endl;
    std::cout << "output: epsxout_n = " << epsxout*gam  << ", epsyout_n = " << epsyout*gam << std::endl;
    std::cout << "output: p_epsx_n = " << p_epsx*gam
	      << ", p_epsy_n = " << p_epsy*gam << "\n" << std::endl;

    std::cout << "output: betaxp = " << betaxp
	      << ", betayp = " << betayp << std::endl;
    std::cout << "output: e_rms(MeV) = " << rmsde*e*1000.0
              << ", rmsct = " << rmsct << "\n" << std::endl;

    std::cout << "Delta E(\%) = 1000*e*(avDE-avDEin) = " << 1000*e*(avDE-avDEin) << " MeV" << std::endl;
    std::cout << "Delta sigma E(\%) = 1000*e*(rmsde-rmsdein) " << 1000*e*(rmsde-rmsdein) << " MeV\n" << std::endl;

    std::cout.precision(4); 
    std::cout << machine << "\t" 
              << avDE*e*1000 << "\t"
              << rmsde*e*1000.0 << "\t" 
              << p_epsx*gam*1.0e6 << "\t"
              << p_epsy*gam*1.0e6 << "\t"
              << rmsct*1000.0 << "\t"
              << epsxout*gam*1.0e6 << "\t"
              << epsyout*gam*1.0e6 << "\t" 
              << std::endl;

    std::ofstream xmgrparms("../out/xmgrace/bunch/bunch.par");
    xmgrparms.precision(3);
    xmgrparms << xmgracestring1 << "\"" << "qtot(C): " << qtot
	    << ",  SYMMETRIZED Np: "     << Np 
	    << ",  Nturns: " << Nturns 
	    << ",  seed: " << seedSave
	    << ",  ee(GeV): " << ee 
	    << ",  lstr(mm): " << lstr*1.0e3
	    << ",  strH(mu): " << strH*1.0e6
	    << ",  LATT: " << machine << "\""
	    << xmgracestring2 << "\""
	    << "xhW(micron): " << xhW*1.0e6
	    << ",  yhW(micron): " << yhW*1.0e6
	    << ",  cthW(mm): " << cthW*1.0e3
	    << ",  dehW(%): " << dehW*1.0e2  
	    << ",  d(1): " << ddeBydct
	    << ",  d(2): " << d2deBydct2
	    << ", " << longit_dist << " longit." 
      // << ", RAT_MAX: " << EPSX_RATIO_MAX
      // << ", number cut: " << number_cut
            << "\"" 
	    << xmgracestring3 << "\""
	    << "IN  :     betax(m): " << betaxin 
	    << ",    betay(m): " << betayin
	    << ",  p_epsx(m): " << p_epsxin
	    << ",  p_epsy(m): " << p_epsyin
	    << ",  de_rms(%): " << rmsdein*1.0e2 
	    << ",  ct_rms(mm): " << rmsctin*1.0e3 << "\""
	    << xmgracestring4 << "\""
	    << "OUT:  betaxp(m): " << betaxp
	    << ",  betayp(m): " << betayp
	    << ",  p_epsx(m): " << p_epsx
	    << ",  p_epsy(m): " << p_epsy 
	    << ",  de_rms(%): " << rmsde*1.0e2
	    << ",  ct_rms(mm): " << rmsct*1.0e3
            << "\"" << std::endl;
  }
}
