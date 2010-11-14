
#include "Shell.h"

using namespace UAL;

int main()
{
  COSY::UAL1::Shell shell;

  // ************************************************************************
  // std::cout << "\nDefine the space of Taylor maps." << std::endl;
  // ************************************************************************

  shell.setMapAttributes(Args() << Arg("order", 5));

  // ************************************************************************
  std::cout << "\nBuild lattice " << std::endl;
  // ************************************************************************

  shell.readSXF(Args() << Arg("file",  "./data/COSY.sxf"));

  // ************************************************************************
  std::cout << "\nAdd split ." << std::endl;
  // ************************************************************************

  shell.addSplit(Args() << Arg("lattice", "cosy") << Arg("types", "Sbend")
  		 << Arg("ir", 4));

  shell.addSplit(Args() << Arg("lattice", "cosy") << Arg("types", "Quadrupole")
  		 << Arg("ir", 4));

  // ************************************************************************
  std::cout << "Select lattice." << std::endl;
  // ************************************************************************

  shell.use(Args() << Arg("lattice", "cosy"));

  // ************************************************************************
  std::cout << "Add K1 and K2." << std::endl;
  // ************************************************************************

  double k1dscale = 1.0;
 

  double beqs [24] = {
     0.00024865454,  // beq01
     0.000290005055, // beg02
     0.000364696551, // beq03
    -4.39938447e-05, // beq04 
    -5.02770335e-05, // beq05
     0.000340023466, // beq06
     0.000218466286, // beq07
     0.000188250485, // beq08
     9.04582382e-05, // beq09
     0.000565246685, // beq10
     0.000446412357, // beq11
    -0.000105023975, // beq12
     8.9633605e-05,  // beq13
     0.000491963725, // beq14
     0.000541445791, // beq15
     0.000190897402, // beq16
     0.000181602083, // beq17
     0.000270748924, // beq18 
     0.000309054199, // beq19
    -0.000102847503, // beq20
     0.000154787272, // beq21
     0.000420139713, // beq22
     0.000571344314, // beq23
    -3.59846136e-05};// beq24

  double k2dscale = 1.0;

 /*
bes01:=-0.0200720795;
bes02:=-0.0212360366;
bes03:=-0.0212942433;
bes04:=-0.0200618945;
bes05:=-0.0195802792;
bes06:=-0.0180427136;
bes07:=-0.0218446443;
bes08:=-0.0170255594;
bes09:=-0.0203181775;
bes10:=-0.0266770387;
bes11:=-0.0187608173;
bes12:=-0.0202064733;
bes13:=-0.0178661488;
bes14:=-0.0202668768;
bes15:=-0.0250611256;
bes16:=-0.0158581336;
bes17:=-0.0229458656;
bes18:=-0.0191868654;
bes19:=-0.00377373102;
bes20:=-0.0224494799;
bes21:=-0.0219379186;
bes22:=-0.0241846285;
bes23:=-0.021817414;
bes24:=-0.0193134543;
 */

  char bendName [5];

  for(int i=1; i < 25; i++){
    sprintf(bendName, "be%d", i);
    shell.addMadK1K2(bendName, beqs[i-1]*k1dscale, 0.0);
  }


  // ************************************************************************
  std::cout << "\nWrite SXF file ." << std::endl;
  // ************************************************************************

  std::string outputFile = "./out/cosy.sxf";

  shell.writeSXF(Args() << Arg("file",  outputFile.c_str()));

  // ************************************************************************
  std::cout << "\nDefine beam parameters." << std::endl;
  // ************************************************************************

  double mass   = 0.9382796; // proton 
  double pc     = 1.85; 
  double energy = sqrt(mass*mass + pc*pc);

  shell.setBeamAttributes(Args() << Arg("energy", energy) 
                                 << Arg("mass", mass));

  PAC::BeamAttributes& ba = shell.getBeamAttributes();

  // ************************************************************************
  std::cout << "\nLinear analysis." << std::endl;
  // ************************************************************************
  
  std::string analysisFile = "./out/cosy.analysis";

  shell.analysis(Args());

  // Calculate a linear matrix

  std::string mapFile = "./out/cosy.map1";

  std::cout << " matrix" << std::endl;
  shell.map(Args() << Arg("order", 1) << Arg("print", mapFile.c_str()));

  // Calculate twiss
  
  std::string twissFile = "./out/cosy.twiss";

  std::cout << " twiss " << std::endl;
  shell.twiss(Args() << Arg("print", twissFile.c_str()));

}
