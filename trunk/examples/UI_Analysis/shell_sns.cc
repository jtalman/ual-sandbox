
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
  
  shell.readADXF(Args() << Arg("file",  "./data/ff_sext_latnat.adxf"));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  shell.addSplit(Args() << Arg("lattice", "ring") << Arg("types", "Sbend")
		 << Arg("ir", 2));

  shell.addSplit(Args() << Arg("lattice", "ring") << Arg("types", "Quadrupole")
		 << Arg("ir", 2));

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "ring"));

  // ************************************************************************
  std::cout << "\nWrite ADXF file ." << std::endl;
  // ************************************************************************
   
  shell.writeADXF(Args() << Arg("file",  "./out/cpp/ff_sext_latnat.adxf"));


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
  std::cout << " twiss (ring )" << std::endl;
  
  shell.twiss(Args() << Arg("print", "./out/cpp/ring.twiss")); 

  std::cout << " twiss (transfer line)" << std::endl;

  PacTwissData tw;
  tw.beta(0, 2.626);
  tw.alpha(0, 0.5705);
  tw.d(0, 1.406e-05);
  tw.dp(0, 0);

  tw.beta(1, 12.32);
  tw.alpha(1, -2.26);

  
  shell.twiss(Args() << Arg("print", "./out/cpp/tline.twiss"), tw); 


}

