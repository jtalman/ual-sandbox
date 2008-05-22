#include <time.h>
#include "Optics/PacTMap.h"

int main()
{

  time_t now;

  int dimension = 6;
  int maxOrder  = 6;

  ZLIB::Space space(dimension, maxOrder);

  // Initialize Da objects

  int size = dimension;

  ZLIB::VTps vtps(size), mult(size);

  ZLIB::Tps tps;

  int someOrder = maxOrder;
  tps.order(someOrder);

  unsigned int i;
  for(i=0; i < tps.size(); i++) { tps[i] = (i + 1.)*0.01; }

  for(i=0; i < vtps.size(); i++) { vtps[i] = (i+1)*tps; }

  // Multiplication

  now = time(NULL); printf("%s\n", ctime(&now));
  mult = vtps * (5*vtps);
  now = time(NULL); printf("%s\n", ctime(&now));

  // Translate DaVTps to PacTMap

  PacTMap map(size);
  map = mult;

  // Beam

  PAC::Position p;
  p.set(1.0e-3, 0.0, 1.0e-3, 0.0, 0.0, 1.0e-5);

  PAC::Bunch bunch(1);
  bunch[0].setPosition(p);

  // Tracking

  int turns = 3;
  map.propagate(bunch, turns); 

  // I/O

  p =  bunch[0].getPosition();  

  ofstream TFile("./out/cxx.new"); 
  TFile << p.getX()  << " " << p.getPX() << " " << p.getY()  << " " 
	<< p.getPY() << " " << p.getCT() << " " << p.getDE() << "\n";
  TFile.close();

}
