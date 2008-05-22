#include "echo_sxf/EchoParser.hh"

// extern yyparse(void);

int main (int argc, char** argv) { 

  if(argc != 3){
    cerr << "Usage: sxf_tester <sxf file> <echo file>" << endl;
    return 0;
  }

  // Initialize the SXF echo parser
  SXF::EchoParser parser;

  // Read data
  parser.read(argv[1], argv[2]);

  return 1;
}
