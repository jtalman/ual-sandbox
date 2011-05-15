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
 std::string apdfFile = "./data/eteapot.apdf";
 std::string orbitFile = "./out/cpp/";
 orbitFile += mysxfbase;
 orbitFile += ".orbit";

 int split = 2;
 int order = 5;
 int turns = 4;
