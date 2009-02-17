
#include "UAL/UI/Shell.hh"

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

  shell.readSXF(Args() << Arg("file",  "./data/proton_ring.sxf"));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  shell.addSplit(Args() << Arg("lattice", "edm") << Arg("types", "Sbend")
		 << Arg("ir", 2));

  shell.addSplit(Args() << Arg("lattice", "edm") << Arg("types", "Quadrupole")
		 << Arg("ir", 2));

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "edm"));

  // ************************************************************************
  std::cout << "\nWrite ADXF file ." << std::endl;
  // ************************************************************************

  shell.writeSXF(Args() << Arg("file",  "./out/cpp/proton_ring.sxf"));


  // ************************************************************************
  std::cout << "\nDefine beam parameters." << std::endl;
  // ************************************************************************

  shell.setBeamAttributes(Args() << Arg("energy", 1.93827231)
			  << Arg("mass", 0.93827231));

  // ************************************************************************
  std::cout << "\nLinear analysis." << std::endl;
  // ************************************************************************

  // Make linear matrix
  std::cout << " matrix" << std::endl;

  shell.map(Args() << Arg("order", 1) << Arg("print", "./out/cpp/map1"));

  // Calculate twiss
  std::cout << " twiss (edm )" << std::endl;

  shell.twiss(Args() << Arg("print", "./out/cpp/edm.twiss"));


}

