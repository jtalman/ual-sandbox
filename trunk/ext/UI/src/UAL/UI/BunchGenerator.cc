
#include "UAL/Common/Def.hh"
#include "ACCSIM/Bunch/BunchGenerator.hh"
#include "Optics/PacChromData.h"

#include "UAL/UI/BunchGenerator.hh"

UAL::BunchGenerator::BunchGenerator()
{
  m_type      = "gauss";

  m_np        = 0;
  m_enx       = m_eny = m_et = 0.0;
  ctHalfWidth = 0;
  deHalfWidth = 0; 

  m_seed   = -1;
}

bool  UAL::BunchGenerator::setBunchArguments(const UAL::Arguments& arguments)
{
  std::map<std::string, UAL::Argument*>::const_iterator it;
  const std::map<std::string, UAL::Argument*>& args = arguments.getMap();

  it = args.find("type");
  if(it != args.end()) m_type = it->second->getString();

  it = args.find("np");
  if(it != args.end()) m_np = (int) it->second->getNumber();

  it = args.find("enx");
  if(it != args.end()) m_enx = it->second->getNumber();

  it = args.find("eny");
  if(it != args.end()) m_eny = it->second->getNumber();

  it = args.find("ctMax");
  if(it != args.end()) ctHalfWidth = it->second->getNumber();

  it = args.find("deMax");
  if(it != args.end()) deHalfWidth = it->second->getNumber();

  it = args.find("et");
  if(it != args.end()) m_et = it->second->getNumber();

  it = args.find("seed");
  if(it != args.end()) m_seed = (int) it->second->getNumber();
  
  return true;
}

void UAL::BunchGenerator::updateBunch(PAC::Bunch& bunch, PacTwissData& twiss)
{

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();

  double m       = ba.getMass();
  double e       = ba.getEnergy();
  double p2      = e*e - m*m;
  double gamma   = e/m;
  double v0byc2  = p2/(e*e);
  double v0byc   = sqrt(v0byc2);

  if(m_type == "grid") {
    std::cout << "grid distribution " << std::endl;
    updateGridBunch(bunch, twiss, v0byc, gamma);
  }
  else {
    std::cout << "gauss distribution " << std::endl;
    updateGaussBunch(bunch, twiss, v0byc, gamma);
  }
}

void UAL::BunchGenerator::updateGridBunch(PAC::Bunch& bunch, 
					  PacTwissData& twiss,
					  double v0byc, 
					  double gamma)
{
  bunch.resize(m_np);
  for(int ip =0; ip < bunch.size(); ip++){
    bunch[ip].setFlag(0);
    bunch[ip].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }

  if(m_np == 0) return;

  // Transverse part

  double ex = m_enx/v0byc/gamma;
  double ey = m_eny/v0byc/gamma;

  double xmax = sqrt(ex*twiss.beta(0));
  double ymax = sqrt(ey*twiss.beta(1));

  int n1 = sqrt(1.*m_np);
  int n2 = m_np - n1*n1;

  double dr = 1./n1;
  double da = UAL::pi/2./n1;

  int counter = 0;
  for(int ir = 0; ir < n1; ir++){
    double r = dr*(ir+1);
    for(int ia = 0; ia < n1; ia++){
      double angle = da*ia;
      double xi = sqrt(r*ex*cos(angle)*twiss.beta(0));
      double yi = sqrt(r*ey*sin(angle)*twiss.beta(1));
      bunch[counter].getPosition().set(xi, 0.0, yi, 0.0, 0.0, 0.0);
      counter++;
    } 
  }

  if (n2 > 0) {
    double r = 1. + dr;
    for(int ia = 0; ia < n2; ia++){
      double angle = da*ia;
      double xi = sqrt(r*ex*cos(angle)*twiss.beta(0));
      double yi = sqrt(r*ey*sin(angle)*twiss.beta(1));
      bunch[counter].getPosition().set(xi, 0.0, yi, 0.0, 0.0, 0.0);
      counter++;
    } 
  }

  // Longitudinal part
  for(int ie = 0; ie < m_np; ie++){
    bunch[ie].getPosition().setDE((deHalfWidth/m_np)*ie);
  } 
}

void UAL::BunchGenerator::updateGaussBunch(PAC::Bunch& bunch, 
					   PacTwissData& twiss,
					   double v0byc, 
					   double gamma)
{
  bunch.resize(m_np);

  for(int ip =0; ip < bunch.size(); ip++){
    bunch[ip].setFlag(0);
    bunch[ip].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }

  ACCSIM::BunchGenerator bunchGenerator;


  // Longitudinal part

  bunchGenerator.addBinomialEllipse1D(bunch, 
				      3,             // m, gauss
				      4,             // ct index
				      ctHalfWidth,   // ct half width
				      5,             // de index
				      deHalfWidth,   // de half width
				      0.0,           // alpha
				      m_seed);       // seed

  // Transverse part

  double ex = m_enx/v0byc/gamma;
  double ey = m_eny/v0byc/gamma;

  double mFactor = 3;

  PAC::Position emittance;
  emittance.set(ex, 0.0, ey, 0.0, 0.0, 0.0);

  bunchGenerator.addBinomialEllipses(bunch, mFactor, 
				     twiss, 
				     emittance, m_seed);

  return;
}

