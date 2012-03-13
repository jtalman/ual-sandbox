   std::string sxfFile = argv[1];
// std::string sxfFile = "./data/";
// sxfFile += argv[2];
// sxfFile += ".sxf";

 std::string outputFile = "./out/cpp/";
 outputFile += mysxfbase;
 outputFile += ".sxf";
 std::string mapFile = "./out/cpp/";
 mapFile += mysxfbase;
 mapFile += ".map1";
 std::string twissFile = "./out/cpp/";
 twissFile += mysxfbase;
 twissFile += ".twiss";
//std::string apdfFile = "./data/eteapotConservedVector.apdf";
//std::string apdfFile = "./data/eteapotLegacyBenchmark.apdf";
  std::string apdfFile = "./data/eteapot.apdf";
 std::string orbitFile = "./out/cpp/";
 orbitFile += mysxfbase;
 orbitFile += ".orbit";

 int split = 1;
 int order = 2;
 int turns;                // specified as 1 in trtrin (for post processing)
                           // might be overwritten tp multiple turns (e.g. 10) in simulatedProbeValues
