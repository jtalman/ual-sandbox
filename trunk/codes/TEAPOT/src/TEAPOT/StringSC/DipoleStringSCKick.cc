
#include "UAL/APF/PropagatorFactory.hh"
#include "PAC/Beam/Bunch.hh"
#include "SMF/PacElemAttributes.h"
#include "TEAPOT/Integrator/MagnetData.hh"
#include "TEAPOT/Integrator/TrackerFactory.hh"
#include "TEAPOT/StringSC/DipoleStringSCKick.hh"
#include "TEAPOT/StringSC/StringSCSolver.hh"

using namespace std;

TEAPOT::DipoleAlgorithm<double, PAC::Position> TEAPOT::DipoleStringSCKick::s_algorithm;

TEAPOT::DipoleStringSCKick::DipoleStringSCKick()
{

  m_ir    = 0.0;

  m_ke1 = 0;
  m_ke2 = 0;
}

TEAPOT::DipoleStringSCKick::DipoleStringSCKick(const TEAPOT::DipoleStringSCKick& p)
{
  m_mdata  = p.m_mdata;
  m_data   = p.m_data;

  m_ke1 = p.m_ke1;
  m_ke2 = p.m_ke2;

  m_ir     = p.m_ir;
}

TEAPOT::DipoleStringSCKick::~DipoleStringSCKick()
{
}

UAL::PropagatorNode* TEAPOT::DipoleStringSCKick::clone()
{
  return new TEAPOT::DipoleStringSCKick(*this);
}

void TEAPOT::DipoleStringSCKick::setLatticeElements(const UAL::AcceleratorNode& sequence, 
						    int is0, int is1, 
						    const UAL::AttributeSet& attSet)
{

   TEAPOT::BasicTracker::setLatticeElements(sequence, is0, is1, attSet);  
   const PacLattice& lattice     = (PacLattice&) sequence;

   setLatticeElement(lattice[is0]);

}

void TEAPOT::DipoleStringSCKick::setLatticeElement(const PacLattElement& e)
{

  m_name  = e.getDesignName();
  m_ir    = e.getN();

  m_data.setLatticeElement(e);

  m_l     = m_data.m_l;
  m_angle = m_data.m_angle;

  // m_mdata.setLatticeElement(e);
  setMagnetData(m_mdata,  e);

}

void TEAPOT::DipoleStringSCKick::setMagnetData(MagnetData& md, const PacLattElement& e)
{
  PacElemBend *bend = 0;
  double e1 = 0.0, e2 = 0.0;

  // Entry multipole
  PacElemAttributes* front  = e.getFront();
  if(front){
     PacElemAttributes::iterator it = front->find(PAC_BEND);
     if(it != front->end()){
       bend = (PacElemBend*) &(*it);
       e1 = bend->angle();
     }
     // if(it != front->end()) md.m_entryMlt = (PacElemMultipole*) &(*it);
  }

  // Exit multipole
  PacElemAttributes* end  = e.getEnd();
  if(end){
     PacElemAttributes::iterator it = end->find(PAC_BEND);
     //    if(it != end->end()) md.m_exitMlt = (PacElemMultipole*) &(*it);
     if(it != end->end()){
       bend = (PacElemBend*) &(*it);
       e2 = bend->angle();
     }
  }

  // Body attributes
  PacElemAttributes* attributes = e.getBody();

  if(attributes){
    for(PacElemAttributes::iterator it = attributes->begin(); it != attributes->end(); it++){
      switch((*it).key()){
      case PAC_MULTIPOLE:
	md.m_mlt = (PacElemMultipole*) &(*it);
	break;
      case PAC_OFFSET:
	md.m_offset = (PacElemOffset*) &(*it);
	break;
      case PAC_APERTURE:
	// m_aperture = (PacElemAperture*) &(*it);
	break;
      case PAC_ROTATION:
	md.m_rotation = (PacElemRotation*) &(*it);
	break;
      default:
	break;
      }
    }
  } 
     
  double rho = m_l/m_angle;
  if(m_l){
       m_ke1 = -sin(e1)/cos(e1)/rho;
       m_ke2 = -sin(e2)/cos(e2)/rho;
  } 
  // std::cout << "ke1 = " <<  m_ke1 << ", ke2 = " << m_ke2 << std::endl;
}

void TEAPOT::DipoleStringSCKick::propagateSimpleElement(PAC::Bunch& bunch, double v0byc)
{

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);
  // double v0byc = p0/e0;

  PAC::Position tmp;

  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

    // s_algorithm.passEntry(m_mdata, p);

    s_algorithm.makeVelocity(p, tmp, v0byc);
    s_algorithm.makeRV(p, tmp, e0, p0, m0);

    s_algorithm.passBendSlice(m_data.m_slices[0], p, tmp, v0byc);
    s_algorithm.applyMltKick(m_mdata, 1, p); 
    s_algorithm.applyThinBendKick(m_data, m_mdata, 1, p, v0byc);

  }

    // string sc kick
    TEAPOT::StringSCSolver& scSolver = TEAPOT::StringSCSolver::getInstance();

    double bendfac = scSolver.getBendfac();

    double L  = m_data.m_l;
    double Ri = m_data.m_angle*bendfac/m_data.m_l;

    int sRi = 1;
    if ( Ri < 0 ) sRi = -1; 

    /*
    std::cout << m_name << " dipole (simple) , L = " << L << ", angle = " << m_data.m_angle 
	      << ",Ri = " << Ri << ", sRi = " << sRi << ", bendfac = " << bendfac << std::endl;
    */


    Ri = fabs(Ri);
    scSolver.propagate(bunch, Ri, sRi, L); 

    e0 = ba.getEnergy();
    m0 = ba.getMass();
    p0 = sqrt(e0*e0 - m0*m0);
    // v0byc = p0/e0;

    for(int ip = 0; ip < bunch.size(); ip++) {
      if(bunch[ip].isLost()) continue;
      PAC::Position& p = bunch[ip].getPosition();
      tmp = p;

      s_algorithm.makeVelocity(p, tmp, v0byc);
      s_algorithm.makeRV(p, tmp, e0, p0, m0);


      s_algorithm.passBendSlice(m_data.m_slices[1], p, tmp, v0byc);

      // s_algorithm.passExit(m_mdata, p);  
    } 

    /*
      std::cout << "after dipole " << m_name << std::endl;
      for(int i =0; i < bunch.size(); i++){
      PAC::Position p = bunch[i].getPosition();
      std::cout << i << " " 
	      << p[0] << " " << p[1] << " " 
	      << p[2] << " " << p[3] << " " 
	      << p[4] << " " << p[5] << std::endl;
	      }
    */

}

void TEAPOT::DipoleStringSCKick::propagateComplexElement(PAC::Bunch& bunch, double v0byc)
{

  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);
  // v0byc = p0/e0;

  PAC::Position tmp;



  double rIr = 1./m_ir;
  double rkicks = 0.25*rIr;

  TEAPOT::StringSCSolver& scSolver = TEAPOT::StringSCSolver::getInstance();
  double bendfac = scSolver.getBendfac();

  // slicies

  int counter = -1;
  for(int i = 0; i < m_ir; i++){
    for(int is = 1; is < 5; is++){
      counter++;
  
      // 1st half of dipole slice + kick
      for(int ip = 0; ip < bunch.size(); ip++) {
	if(bunch[ip].isLost()) continue;
	PAC::Position& p = bunch[ip].getPosition();
	tmp = p;

	s_algorithm.makeVelocity(p, tmp, v0byc);
	s_algorithm.makeRV(p, tmp, e0, p0, m0);
	s_algorithm.passBendSlice(m_data.m_slices[counter], p, tmp, v0byc);
	s_algorithm.applyMltKick(m_mdata, rkicks, p); 
	s_algorithm.applyThinBendKick(m_data, m_mdata, rkicks, p, v0byc);
     
      }

      // string sc kick

      double L  = m_data.m_l;
      double Ri = m_data.m_angle*bendfac/m_data.m_l;

      int sRi = 1;
      if ( Ri < 0 ) sRi = -1; 

      /*
      std::cout << m_name << " dipole(complex) , L = " << L*rkicks << ", angle = " << m_data.m_angle*rkicks 
	      << ",Ri = " << Ri << ", sRi = " << sRi << ", bendfac = " << bendfac << std::endl;
      */

      Ri = fabs(Ri);
      scSolver.propagate(bunch, Ri, sRi, L*rkicks); 

      e0 = ba.getEnergy();
      m0 = ba.getMass();
      p0 = sqrt(e0*e0 - m0*m0);
      // v0byc = p0/e0;
      
    }
      
    counter++;

    // 
    for(int ip = 0; ip < bunch.size(); ip++) {
      if(bunch[ip].isLost()) continue;
	PAC::Position& p = bunch[ip].getPosition();
	tmp = p;
	s_algorithm.makeVelocity(p, tmp, v0byc);
	s_algorithm.makeRV(p, tmp, e0, p0, m0);
	s_algorithm.passBendSlice(m_data.m_slices[counter], p, tmp, v0byc);
    }

  }



}

    /* 
    3 June, 2006. Inverse bend radius Ri is retained as positive value Ri=|Ri|
    and the bend orientation is maintained by (sign) sRi. Note though that 
    sRi=1 when Ri=0. Bend angles, with their correct algebraic signs, are entered  
    via instructions such as 
              scSolver.setAngle("k2", -0.0163625);
    in "main.cc".
    */

void TEAPOT::DipoleStringSCKick::propagate(UAL::Probe& probe)
{
  PAC::Bunch& bunch = static_cast<PAC::Bunch&>(probe);
  
  PAC::BeamAttributes& ba = bunch.getBeamAttributes();
  double e0 = ba.getEnergy(), m0 = ba.getMass();
  double p0 = sqrt(e0*e0 - m0*m0);
  double v0byc = p0/e0;

  double oldT = ba.getElapsedTime();

  PAC::Position tmp;

  // Entry
  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

    // s_algorithm.passEntry(m_mdata, p);
    p[1] -= m_ke1*p[0]; 
    p[3] += m_ke1*p[2]; 
    
    // s_algorithm.makeVelocity(p, tmp, v0byc);
    // s_algorithm.makeRV(p, tmp, e0, p0, m0);
  }

  if(!m_ir) propagateSimpleElement(bunch, v0byc);
  else propagateComplexElement(bunch, v0byc);


  // Exit
  for(int ip = 0; ip < bunch.size(); ip++) {
    if(bunch[ip].isLost()) continue;
    PAC::Position& p = bunch[ip].getPosition();
    tmp = p;

    // s_algorithm.passExit(m_mdata, p);
      p[1] -= m_ke2*p[0]; 
      p[3] += m_ke2*p[2]; 
  }

  checkAperture(bunch);

  ba.setElapsedTime(oldT + m_l/v0byc/UAL::clight);  

}




TEAPOT::DipoleStringSCKickRegister::DipoleStringSCKickRegister()
{
  UAL::PropagatorNodePtr nodePtr(new TEAPOT::DipoleStringSCKick());
  UAL::PropagatorFactory::getInstance().add("TEAPOT::DipoleStringSCKick", nodePtr);
}



static TEAPOT::DipoleStringSCKickRegister theTeapotDipoleStringSCKickRegister; 
