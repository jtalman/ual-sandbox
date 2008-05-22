#include <iostream>
#include <jni.h>
#include <unistd.h>
#include "timer.h"

#include "UAL/JVM/JWindowManager.hh"

#include "UAL/Common/Def.hh"
#include "UAL/APDF/APDF_Builder.hh"
#include "SMF/PacSmf.h"
#include "Main/Teapot.h"
#include "ual_sxf/Parser.hh"
#include "ACCSIM/Bunch/BunchGenerator.hh"

#include "Shell.hh"

GT::Shell::Shell()
{
  init();
  
  m_ap = 0;
  m_rfTracker = 0;
}

GT::Shell::~Shell()
{
}

void GT::Shell::readInputFile(const char* fileName)
{
  // Open file 

  ifstream file;
  file.open(fileName);
  if(!file) {
    std::cerr << "Cannot open TIBETAN input file " << fileName << std::endl;
    return;
  }

  // Read data

  file >> rauschen >> if_rausch;
  file >> ifcoup >> ifhigh >> ifjmp >> iffdk;
  file >> n >> aconst >> scale >> iran;
  file >> z >> a >> h >> nvmax >> phis >> r0 >> prfstp;

  int i;
  vr.resize(nvmax);
  for(i = 0; i < nvmax; i++) file >> vr[i];

  tr.resize(nvmax);
  for(i = 0; i < nvmax; i++) file >> tr[i];

  file >> gami >> gamf >> nout >> ifpsw >> ifbuck >> ibuc;
  file >> ikin >> dt0  >> pmin >> pone  >> ptwo >> pmax >> detr;
  file >> dt1 >> dt2;
  file >> inmode >> inmode2 >> ds0 >> ph0 >> ds1;
  file >> ifcut >> ncut >> ifcoun >> delap >> tranap >> ifprin;   // 11th line
  file >> ifpj >> npjscl >> ndtgap >> ifot3 >> ifot4;             // 12th line
  file >> gt0 >> gtsw >> tswit >> ifphis;                         // 13th line
  file >> dgt;                                                    // 14th line

  file.close();  
}

void GT::Shell::readSXFFile(const char* sxfFile, const char* accName, 
			    const char* outDirName)
{

  string echoFile = outDirName;
  echoFile += "/";
  echoFile += accName;
  echoFile += ".sxf.echo";
  string outFile =  outDirName;
  outFile += "/";
  outFile += accName;
  outFile += ".sxf.out";

  UAL_SXF_Parser parser;
  parser.read(sxfFile, echoFile.data()); 
  parser.write(outFile.data());

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " << accName << " accelerator " << endl;
    exit(1);
  }

  PacLattice lattice = *latIterator;

  Teapot teapot(lattice);
  PacSurveyData survey;
  teapot.survey(survey);
  double suml = survey.survey().suml();
  
  // std::cout << "lattice length: " << suml << " [m]" << std::endl;
  // std::cout << "r0 from sxf   file " << suml/2./UAL::pi << std::endl;
  // std::cout << "r0 from input file " << r0 << std::endl;

  r0 = suml/2./UAL::pi;

}

void GT::Shell::readAPDFFile(const char* apdfFile)
{

  // Open file 

  ifstream file;
  file.open(apdfFile);
  if(!file) {
    std::cerr << "Cannot open the APDF  file " << apdfFile << std::endl;
    return;
  }
  file.close();

  PAC::BeamAttributes ba;
  ba.setMass(a*UAL::pmass);

  UAL::APDF_Builder apBuilder; 
  apBuilder.setBeamAttributes(ba);

  std::string xmlFile = apdfFile;
  m_ap = apBuilder.parse(xmlFile);

  if(m_ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return;
  }

  double gt = getGT();
  std::cout << "gt = " << gt << std::endl;
  gt0  = gt;
  gtsw = gt;


  double alpha1 = getAlpha1(); 
  dgt  = alpha1;
  std::cout << "alpha1 = " << alpha1 << std::endl; 
  

}

void GT::Shell::selectRFCavity(const char* rfName)
{
  if(m_ap == 0) return;

  UAL::PropagatorNodePtr rfNodePtr;
  UAL::PropagatorSequence& rootNode = m_ap->getRootNode();
  for(UAL::PropagatorIterator pit = rootNode.begin(); pit != rootNode.end(); pit++){
    if((*pit)->getFrontAcceleratorNode().getDesignName() == rfName) rfNodePtr = *pit;
  }

  m_rfTracker = (TIBETAN::RFCavityTracker*) rfNodePtr.operator->(); 
  m_rfTracker->setRF(vr[0]*1.e-9, h,(asin(detr*1.e+6/vr[0]))/(2.*UAL::pi));

}

void GT::Shell::initBunch(double ctHalfWidth, double deHalfWidth)
{

  m_bunch.resize(n);
  m_bunch.setMass(a*UAL::pmass);
  m_bunch.setEnergy(a*UAL::pmass*gami);

  double e = m_bunch.getEnergy(), m = m_bunch.getMass();
  double v0byc = sqrt(e*e - m*m)/e;
  double suml = 2*UAL::pi*r0;
  m_bunch.setRevfreq(v0byc*UAL::clight/suml);

  // std::cout << "Bunch revolution frequency: " << m_bunch.getRevfreq() << std::endl; 
  // std::cout << "T: " << 1./m_bunch.getRevfreq() << " [sec]" << std::endl;  

  ACCSIM::BunchGenerator bunchGenerator;
  int   seed = iran;
  bunchGenerator.addBinomialEllipse1D(m_bunch, 
				      3,             // m, gauss
				      4,             // ct index
				      ctHalfWidth,   // ct half width
				      5,             // de index
				      deHalfWidth,   // de half width
				      0.0,           // alpha
				      seed);         // seed

}


void GT::Shell::track()
{
  double t; // time variable
  
  double suml = 2.0*UAL::pi*r0;
  m_wcMonitor.setBins(npjscl, -suml/h/2.0, suml/h/2.0); 

  int isJumped = 0;
 
  int turn = 0;
  int lost = 0;
  double gamma = m_bunch.getEnergy()/m_bunch.getMass();
  double elapsedTime = 0.0;
 
  start_ms();  
  std::cout << " phase before = " << 360.*(asin(detr*1.e+6/vr[0]))/(2.*UAL::pi) << std::endl;
  std::cout << " phase after = " << 360.*(UAL::pi - asin(detr*1.e+6/vr[0]))/(2.*UAL::pi) << std::endl;

  while(gamma < gamf){

  // while(turn < 0) {

     if((turn/1000)*1000 == turn) {
       t = (end_ms());
       std::cout << "turn = " << turn << ", time  = " << t << " ms" << endl;
     }   

    // print data
    if((turn/ndtgap)*ndtgap == turn) showProfile(m_bunch, elapsedTime);

    // propagate
    m_ap->propagate(m_bunch);

    // perform momentum scraping
    if((turn/ncut)*ncut == turn) {
      lost = performMomentumScraping(m_bunch);
      if(lost != 0) std::cout << "total lost particles = " << lost << std::endl;
      if(lost == m_bunch.size()) break;
    }

    elapsedTime += 1./m_bunch.getRevfreq(); 
    gamma = m_bunch.getEnergy()/m_bunch.getMass();

    if(ifjmp){
      if(isJumped == 0 && (gamma - gtsw) > 0){
	 m_rfTracker->setRF(vr[0]*1.e-9, h,(UAL::pi - asin(detr*1.e+6/vr[0]))/(2.*UAL::pi));
	 isJumped = 1;
      }
    }

    turn++;    
    
  }

}

void GT::Shell::showProfile(PAC::Bunch& bunch, double elapsedTime)
{
      double gamma = bunch.getEnergy()/bunch.getMass();

      m_wcMonitor.propagate(bunch);
      const std::vector<double>& rho  = m_wcMonitor.getProfile();
      const std::vector<double>& bins = m_wcMonitor.getBins();

      m_profileProxy.addProfile(bins, rho);
      m_profileProxy.updateWindow();
      
      m_rampProxy.addGamma(elapsedTime, gamma);
      m_rampProxy.addGammaT(elapsedTime, gt0);
      m_rampProxy.updateData();

}

int GT::Shell::performMomentumScraping(PAC::Bunch& bunch)
{
  int lost = 0;
  for(int ip = 0; ip < bunch.size(); ip++){
    if(bunch[ip].isLost()) {
      lost++;
      continue;
    }
    PAC::Position& p = bunch[ip].getPosition();
    if(fabs(p.getDE()) > delap) {
      bunch[ip].setFlag(1);
      lost++;
    }
  } 
  return lost;
}


double GT::Shell::getGT()
{
  PAC::Bunch bunch(1);
  bunch.setRevfreq(UAL::clight/(2.*UAL::pi*r0));
  bunch[0].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0, 1.e-5);

  m_ap->propagate(bunch);

  PAC::Position& p = bunch[0].getPosition();
  double alpha = (p.getCT()/UAL::clight)*bunch.getRevfreq()/p.getDE();
  double gt = 1/sqrt(abs(alpha));

  return gt;
}

double GT::Shell::getAlpha1()
{
  PAC::Bunch bunch(3);
  bunch.setRevfreq(UAL::clight/(2.*UAL::pi*r0));
  bunch[0].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0,  1.e-5);
  bunch[1].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0,  1.e-2);
  bunch[2].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0, -1.e-2);

  m_ap->propagate(bunch);

  double ct0 =  bunch[0].getPosition().getCT();
  double de0 =  bunch[0].getPosition().getDE();

  double alpha0 = (-ct0/UAL::clight)*bunch.getRevfreq()/de0;  
  // std::cout << "alpha0 = " << alpha0 << std::endl;

  double ct1 =  bunch[1].getPosition().getCT();
  double de1 =  bunch[1].getPosition().getDE();

  double alpha1 = (-ct1/UAL::clight)*bunch.getRevfreq() - alpha0*de1;
  // std::cout << "alpha1 (1) = " << alpha1 << std::endl;
  alpha1 /= alpha0*de1*de1;
  // std::cout << "alpha1     = " << alpha1 << std::endl;

  double ct2 =  bunch[2].getPosition().getCT();
  double de2 =  bunch[2].getPosition().getDE();

  alpha1 = (-ct2/UAL::clight)*bunch.getRevfreq() - alpha0*de2;
  // std::cout << "alpha1 (1) = " << alpha1 << std::endl;
  alpha1 /= alpha0*de2*de2;
  // std::cout << "alpha1     = " << alpha1 << std::endl;

  return alpha1;
}

/*
void GT::Shell::updateRF(double t)
{
  int it = m_rfLagTimes.size() - 1;
  for(int i=0; i < m_rfLagTimes.size(); i++){
    if(t < m_rfLagTimes[i]) {
      it = i;
      break;
    }
  }
  double rfLag = (m_rfLagValues[it] -  m_rfLagValues[it-1]);
  rfLag *= (t - m_rfLagValues[it-1])/(m_rfLagValues[it] -  m_rfLagValues[it-1]);
  
  m_rfTracker->setRF(vr[0]*1.e-9, h, rflag);  
}
*/

void GT::Shell::openWindow()
{

  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();

  JavaVMOption options[2];
  options[0].optionString = "-Djava.compiler=NONE";

  string path = "-Djava.class.path=";
  path += "/home/ual/tasks/gt/linux/java";
  path += ":/home/ual/ual1/codes/UAL/lib/java/ualcore.jar";
  path += ":/home/ual/ual1/tools/java/lib/ext/jcommon-0.7.2.jar";
  path += ":/home/ual/ual1/tools/java/lib/ext/jfreechart-0.9.6.jar";
  options[1].optionString = (char*) path.data();

  js->create(options, 2);

  UAL::JWindowManager wm;
  wm.initMainWindow();
  wm.showMainWindow();

  m_profileProxy.initWindow();
  m_profileProxy.showWindow();

  m_rampProxy.initWindow();
  m_rampProxy.setGammaRange(gami, gamf);
  m_rampProxy.showWindow();


}

void GT::Shell::closeWindow()
{
}

void GT::Shell::init()
{
  // 1st line
  rauschen  = 0;
  if_rausch = 0;

  // 2nd line
  ifcoup = 0;
  ifhigh = 0;
  ifjmp  = 0;
  iffdk  = 0;

  // 3rd line
  n       = 0;
  aconst  = 0.0;
  scale   = 0.1;
  iran    = -100;

  // 4th line
  z      = 1;
  a      = 1;
  h      = 0;
  nvmax  = 0;
  phis   = 0.0;
  r0     = 0.0;
  prfstp = 0;

  // 5th and 6th lines are vr and tr vectors

  // 7th line
  gami   = 0.0;
  gamf   = 0.0;
  nout   = 0;
  ifpsw  = 0;
  ifbuck = 0;
  ibuc   = 0;

  // 8th line
  ikin   = 0;
  dt0    = 0.0;
  pmin   = 0.0;
  pone   = 0.0;
  ptwo   = 0.0;
  pmax   = 0.0;
  detr   = 0.0;

  // 9th line
  dt1    = 0.0;
  dt2    = 0.0;

  // 10th line
  inmode  = 1;
  inmode2 = 0; 
  ds0     = 0.0;
  ph0     = 0.0;
  ds1     = 0.0;

  // 11th line
  ifcut   = 0;
  ncut    = 0;
  ifcoun  = 0;
  delap   = 0.1;
  tranap  = 0.0;
  ifprin  = 0;

  // 12th line
  ifpj    = 0;
  npjscl  = 0;
  ndtgap  = 0;
  ifot3   = 0;
  ifot4   = 0;

  // 13th line
  gt0     = 0.0;
  gtsw    = 0.0;
  tswit   = 0.0;
  ifphis  = 0;

  // 14th line
  dgt     = 0.0;

}

