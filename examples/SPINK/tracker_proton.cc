#include <iostream>
#include <fstream>
#include <iomanip>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"

#include "SPINK/Propagator/DipoleErTracker.hh"
#include "SPINK/Propagator/SpinTrackerWriter.hh"

#include "timer.h"

using namespace UAL;

int main(){

  UAL::Shell shell;

  // ************************************************************************
  std::cout << "\nDefine the space of Taylor maps." << std::endl;
  // ************************************************************************

  shell.setMapAttributes(Args() << Arg("order", 5));


  // ************************************************************************
  std::cout << "\nBuild lattice." << std::endl;
  // ************************************************************************

  shell.readSXF(Args() << Arg("file",  "./data/proton_Efield.sxf"));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  shell.addSplit(Args() << Arg("lattice", "proton") << Arg("types", "Sbend")
		 << Arg("ir", 32));

  shell.addSplit(Args() << Arg("lattice", "proton") << Arg("types", "Elseparator")
		 << Arg("ir", 16));

  shell.addSplit(Args() << Arg("lattice", "proton") << Arg("types", "Quadrupole")
		 << Arg("ir", 32));

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "proton"));

  // ************************************************************************
  std::cout << "\nWrite ADXF file ." << std::endl;
  // ************************************************************************

  shell.writeSXF(Args() << Arg("file",  "./out/cpp/proton_Efield.sxf"));

  // ************************************************************************
  std::cout << "\nDefine beam parameters." << std::endl;
  // ************************************************************************

  double cc  = 2.99792458E+8;

  double energy = 1.171064622;
  double mass   = 0.938272029;            //       proton mass [GeV]
  double charge = 1.0;

  shell.setBeamAttributes(Args() << Arg("energy", energy) << Arg("mass", mass)
			  << Arg("charge",charge));

  // ************************************************************************
  std::cout << "\nLinear analysis." << std::endl;
  // ************************************************************************

  // Make linear matrix
  std::cout << " matrix" << std::endl;
  shell.map(Args() << Arg("order", 1) << Arg("print", "./out/cpp/map1"));

  // Calculate twiss
  std::cout << " twiss (proton)" << std::endl;
  shell.twiss(Args() << Arg("print", "./out/cpp/proton_Efield.twiss"));

  std::cout << " calculate suml" << std::endl;
  shell.analysis(Args());

  // ************************************************************************
  std::cout << "\n Beam Part." << std::endl;
  // ************************************************************************

  PAC::BeamAttributes& ba = shell.getBeamAttributes();
  ba.setG(1.7928456);         // proton G factor

  // ************************************************************************
  std::cout << "\n Algorithm Part. " << std::endl;
  // ************************************************************************

  std::string xmlFile = "data/spink_proton.apdf";

  UAL::APDF_Builder apBuilder;

  apBuilder.setBeamAttributes(ba);

  UAL::AcceleratorPropagator* ap = apBuilder.parse(xmlFile);

  if(ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return 1;
  }

  // ************************************************************************
  std::cout << "\n Electric field. " << std::endl;
  // ************************************************************************

  double ER = 0.01677265736, EV = 0.0, EL = 0.0; // GV/m
  double pc = sqrt(energy*energy - mass*mass);
  
  SPINK::DipoleErTracker::setER(ER);
  SPINK::DipoleErTracker::setEV(EV);
  SPINK::DipoleErTracker::setEL(EL);

 // ************************************************************************
  std::cout << "\n Tracking. " << std::endl;
  // ************************************************************************

  string accName = "proton";

  PacLattices::iterator latIterator = PacLattices::instance()->find(accName);
  if(latIterator == PacLattices::instance()->end()){
    std::cerr << "There is no " + accName << " accelerator " << endl;
    return 1;
  }

  PacLattice lattice = *latIterator;

  double t; // time variable

  PAC::Bunch bunch(1);
  bunch.setBeamAttributes(ba);

  PAC::Spin spin;
  spin.setSX(0.0);
  spin.setSY(0.0);
  spin.setSZ(1.0);

  int ip;

  // 3.2 Spink tracker

  // define the intial distribution for your application

  std::ofstream out1("orbit.dat");
  std::ofstream out2("spin.dat");

//  ifstream inFile("pop.dat");

  char endLine = '\0';
  char line1[200];
  char line2[200];

  int turns = 30000;

  SPINK::SpinTrackerWriter* stw = SPINK::SpinTrackerWriter::getInstance();
  
  stw->setFileName("spin_finestep.dat");

  //  bunch[0].getPosition().set( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 );
  bunch[0].getPosition().set( 0.0,  1.0E-3,  0.0,  0.0,  0.0,  0.0 );
  //  bunch[0].getPosition().set( -3.63608948E-04, 0.0,  0.0,  0.0,  0.0,  0.0 );
  //  bunch[0].getPosition().set( 0.0,  0.0,  0.0,  1.0E-3,  0.0,  0.0 );

  for(ip=0; ip < bunch.size(); ip ++){
    //    bunch[ip].getPosition().set(0.0, 0.0, 0.0, 1.0E-3, 0.0, 0.0);
    //    bunch[ip].getPosition().set(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    //    bunch[ip].getPosition().set(0.0, 0.0, 1.0E-3, 0.0, 0.0, 0.0);
    bunch[ip].setSpin(spin);
  }

  std::cout << "\nSpink tracker " << endl;
  std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

  double gam = energy/mass;
  double p   = sqrt(energy*energy - mass*mass);
  double v   = p/gam/energy*UAL::clight;

  // length of accelerator

  double suml = OpticsCalculator::getInstance().suml;
  std::cout << "suml = " << suml << std::endl;

  double wp_time = 0.0;
  double syn_revfreq = v/suml;

  PAC::BeamAttributes& bba = bunch.getBeamAttributes();

  bba.setElapsedTime(wp_time);
  bba.setRevfreq(syn_revfreq);

  /*
  start_ms();
  for(int it = 1; it <= 10; it++){
    ap->propagate(bunch);
  }
  
  t = (end_ms());
  std::cout << "spink tracker' time  = " << t << " ms" << endl;
  */

  for(int iturn = 1; iturn <= turns; iturn++){

    // print global time

    double t0 = bunch.getBeamAttributes().getElapsedTime();
    double revfreq0 = bunch.getBeamAttributes().getRevfreq();

    //    std::cout << "global time: " << t0 << ", rev freq: " << revfreq0 <<  std::endl;

    ap -> propagate(bunch);
    
    for(ip=0; ip < bunch.size(); ip += 1){

      PAC::Position& pos = bunch[ip].getPosition();

      double sx = bunch[ip].getSpin()->getSX();
      double sy = bunch[ip].getSpin()->getSY();
      double sz = bunch[ip].getSpin()->getSZ();
      double x  = pos.getX();
     double px = pos.getPX();
      double y  = pos.getY();
      double py = pos.getPY();
      double ct = pos.getCT();
      double de = pos.getDE();

      double wp_time = t0 + (-ct / cc );

      double spin_g2 = (sx*px+sy*py+sz*(1.0+x/25.0))/sqrt(sx*sx+sy*sy+sz*sz)/sqrt(px*px+py*py+(1.0+x/25.0)*(1.0+x/25.0));


      sprintf(line1, "%1d %7d    %-15.9e %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %c", 
	      ip,iturn,wp_time,x,px,y,py,ct,de,endLine);

      sprintf(line2, "%1d %7d    %-15.9e %-16.7e %-16.7e %-16.7e %-16.7e %c", 
	      ip,iturn,wp_time,sx,sy,sz,spin_g2,endLine);

      out1 << line1 << std::endl;
      out2 << line2 << std::endl;

    }
  }
  
  out1.close();
  out2.close();

  return 1;
}

