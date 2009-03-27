#include <iostream>
#include <fstream>

#include "UAL/APDF/APDF_Builder.hh"
#include "PAC/Beam/Position.hh"
#include "SMF/PacSmf.h"
#include "PAC/Beam/Bunch.hh"
#include "Main/Teapot.h"
#include "UAL/UI/Shell.hh"

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

  shell.readSXF(Args() << Arg("file",  "./data/muon.sxf"));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  shell.addSplit(Args() << Arg("lattice", "muon") << Arg("types", "Sbend")
		 << Arg("ir", 4));

  shell.addSplit(Args() << Arg("lattice", "muon") << Arg("types", "Quadrupole")
		 << Arg("ir", 4));

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "muon"));

  // ************************************************************************
  std::cout << "\nWrite ADXF file ." << std::endl;
  // ************************************************************************

  shell.writeSXF(Args() << Arg("file",  "./out/cpp/muon.sxf"));

  // ************************************************************************
  std::cout << "\nDefine beam parameters." << std::endl;
  // ************************************************************************

  double energy = 0.145477474;
  double mass   = 0.10565839;

  shell.setBeamAttributes(Args() << Arg("energy", energy) << Arg("mass", mass));

  // ************************************************************************
  std::cout << "\nLinear analysis." << std::endl;
  // ************************************************************************

  // Make linear matrix
  std::cout << " matrix" << std::endl;
  shell.map(Args() << Arg("order", 1) << Arg("print", "./out/cpp/map1"));

  // Calculate twiss
  std::cout << " twiss (muon )" << std::endl;
  shell.twiss(Args() << Arg("print", "./out/cpp/muon.twiss"));

  // std::cout << " calculate suml" << std::endl;
  // shell.analysis(Args());

  // ************************************************************************
  std::cout << "\n Beam Part." << std::endl;
  // ************************************************************************

  PAC::BeamAttributes& ba = shell.getBeamAttributes();
  ba.setG(0.0011659230);

  // ************************************************************************
  std::cout << "\n Algorithm Part. " << std::endl;
  // ************************************************************************

  std::string xmlFile = "data/spink.apdf";

  UAL::APDF_Builder apBuilder;

  apBuilder.setBeamAttributes(ba);

  UAL::AcceleratorPropagator* ap = apBuilder.parse(xmlFile);

  if(ap == 0) {
    std::cout << "Accelerator Propagator has not been created " << std::endl;
    return 1;
  }

 // ************************************************************************
  std::cout << "\n Tracking. " << std::endl;
  // ************************************************************************

  string accName = "muon";

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

  // 3.1 Teapot Tracker

  // define the intial distribution for your application

  for(ip=0; ip < bunch.size(); ip++){
    bunch[ip].getPosition().set(0.0, 0.0, 0.0, 1.0E-03, 0.0, 0.0);
    bunch[ip].setSpin(spin);
  }

  std::cout << "\nTeapot Tracker " << endl;
  std::cout << "size : " << lattice.size() << " elements " <<  endl;

  Teapot teapot(lattice);

  start_ms();

  for(int iturn = 1; iturn <= 100; iturn++){
    teapot.track(bunch, 0, lattice.size());
  }

  t = (end_ms());

  std::cout << "teapot tracker's time  = " << t << " ms" << endl;

  for(ip=0; ip < bunch.size(); ip += 1){
    PAC::Position& pos = bunch[ip].getPosition();
    std::cout << ip
	      << " : x = " << pos.getX()
	      << ", px = " << pos.getPX()
	      << ", y = "  << pos.getY()
	      << ", py = " << pos.getPY() << endl;
  }

  // 3.2 Spink tracker

  // define the intial distribution for your application

  std::ofstream out1("orbit.dat");
  std::ofstream out2("spin.dat");

  char endLine = '\0';
  char line1[200];
  char line2[200];

  int turns = 1; // 5000

  for(ip=0; ip < bunch.size(); ip ++){
    bunch[ip].getPosition().set(0.0, 0.0, 0.0, 1.0E-3, 0.0, 0.0);
    bunch[ip].setSpin(spin);
  }

  std::cout << "\nSpink tracker " << endl;
  std::cout << "size : " << ap->getRootNode().size() << " propagators " << endl;

  // set wp time

  double wp_time = 0.0;
  bunch.getBeamAttributes().setElapsedTime(wp_time);

  double p  = sqrt(energy*energy - mass*mass);
  double v = p/energy*UAL::clight;

  // length of accelerator
  double suml = OpticsCalculator::getInstance().suml;
  std::cout << "suml = " << suml << std::endl;

  for(int iturn = 1; iturn <= turns; iturn++){

    start_ms();
  for(int it = 1; it <= 100; it++){
    ap->propagate(bunch);
  }

    t = (end_ms());
    std::cout << "spink tracker' time  = " << t << " ms" << endl;

    // print global time

    wp_time = bunch.getBeamAttributes().getElapsedTime();

    std::cout << "global time: " << wp_time << std::endl;
    
    for(ip=0; ip < bunch.size(); ip += 1){

      PAC::Position& pos = bunch[ip].getPosition();

      std::cout << " bunch " << ip           << ", turn  " << iturn
		<< ": x = " << pos.getX()    << ", px = " << pos.getPX()
		<< ", y =  " << pos.getY()   << ", py = " << pos.getPY()
		<< ", cdt =  " << pos.getCT()<< ", de = " << pos.getDE()
		<< ", sx = " << bunch[ip].getSpin()->getSX()
		<< ", sy = " << bunch[ip].getSpin()->getSY()
		<< ", sz = " << bunch[ip].getSpin()->getSZ() << endl;

      sprintf(line1, "%1d %7d    %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %-15.7e %c", 
	      ip,iturn,pos.getX(),pos.getPX(),pos.getY(),pos.getPY(),pos.getCT(),pos.getDE(),endLine);

      sprintf(line2, "%1d %7d    %-16.7e %-16.7e %-16.7e %-16.7e %-16.7e %c", 
	      ip,iturn,bunch[ip].getSpin()->getSX(),bunch[ip].getSpin()->getSY(),
	      bunch[ip].getSpin()->getSZ(),endLine);

      out1 << line1 << std::endl;
      out2 << line2 << std::endl;

    }
  }

  out1.close();
  out2.close();

  return 1;
}

