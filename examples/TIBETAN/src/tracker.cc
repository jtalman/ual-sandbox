
#include <math.h>
#include <unistd.h>

#include "ZLIB/Tps/Space.hh"

#include "Shell.hh"

int main(){


  ZLIB::Space s_space(6, 5);

  GT::Shell shell;
 
  std::cout << "0. Read TIBETAN input file. " << std::endl;
  shell.readInputFile("/home/ual/tasks/gt/data/tibetan.in");

  std::cout << "1. Open window." << std::endl;
  shell.openWindow();

  std::cout << "2. Read SXF file." << std::endl;
  shell.readSXFFile("/home/ual/tasks/gt/data/blue.sxf", "blue", "/home/ual/tasks/gt/linux/out");

  std::cout << "3. Read APDF file." << std::endl;
  shell.readAPDFFile("/home/ual/tasks/gt/data/tibetan.apdf");

  std::cout << "4. Select RF cavity tracker." << std::endl;  
  shell.selectRFCavity("rfac1");

  std::cout << "5. Make bunch distribution." << std::endl;
  shell.initBunch(1.0, 5.e-3);

  std::cout << "6. Tracking. " << std::endl;
  shell.track();

  while(1){
      sleep(120);
  }

  return 1;

}
