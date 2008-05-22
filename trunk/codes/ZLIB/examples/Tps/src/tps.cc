#include <time.h>
#include "ZLIB/Tps/Tps.hh"

main()
{ 

 unsigned int i;
 time_t now;

 int dimension = 6;
 int maxOrder  = 6;
 
 ZLIB::Space space(dimension, maxOrder);

 // Tps
 
 ZLIB::Tps tps, copy, sum, diff, mult, div, tdiv, sqroot, tsqroot;

 // Initialize

 int someOrder = maxOrder;
 
 tps.order(someOrder);
 for(i=0; i < tps.size(); i++) { tps[i] = (i + 1.)*0.01; }

 // Copy

 copy = tps;

 // Addition

 sum = copy + tps + 1.0;

 // Subtraction

 diff = copy - tps - 2.0;

 // Multiplication

 now = time(NULL); printf("%s\n", ctime(&now));
 for(i = 0; i < 2000; i++) mult = sum*sum;
 now = time(NULL); printf("%s\n", ctime(&now));

 // Division

 div = 1./(1. + mult);
 tdiv = div*(1. + mult);

 // Sq. root

 sqroot  = ZLIB::sqrt(1. + mult);
 tsqroot = 1. + mult - sqroot*sqroot;

 // I/0

 char s[120];
 ofstream tpsFile("./out/tps_cxx.new");
 
 for(i=0; i < tps.size(); i++){
   sprintf(s, "%3d %- 12.6e %- 12.6e %- 12.6e",
	        i, mult[i], tdiv[i], tsqroot[i]);
   tpsFile << s << "\n";
 }

 tpsFile.close();
 
 return 1;
}
