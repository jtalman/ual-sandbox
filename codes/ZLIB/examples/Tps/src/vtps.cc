#include <time.h>
#include "ZLIB/Tps/VTps.hh"

main()
{

  time_t now;

  int dimension = 6;
  int maxOrder  = 6;

  ZLIB::Space space(dimension, maxOrder);

  // VTps

  int size = dimension;

  ZLIB::VTps vtps(size), copy(size), sum(size), diff(size), mult(size);

  // Initialize

  ZLIB::Tps tps;

  int someOrder = maxOrder;
  tps.order(someOrder);

  unsigned int i;
  for(i=0; i < tps.size(); i++) { tps[i] = (i + 1.)*0.01; }
  for(i=0; i < vtps.size(); i++) { vtps[i] = (i+1)*tps; }

  // Copy

  copy = vtps;

  // Addition
  
  sum = copy + vtps + 1.0;

  // Subtraction

  diff = copy - vtps - 2.0;

  // Multiplication

  now = time(NULL); printf("%s\n", ctime(&now));
  mult = vtps * (5*vtps);
  now = time(NULL); printf("%s\n", ctime(&now));

  // I/0

  mult.write("./out/vtps_cxx.new");

  return 1;

}
