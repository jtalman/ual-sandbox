#include "ual_sxf/Parser.hh"

int main (int argc, char** argv) { 

  if(argc != 2){
    cerr << "Usage: parser <file in the ../in directory>" << endl;
    return 0;
  }

  string inFile   = "../in/";  inFile   += argv[1];
  string echoFile = "./echo/"; echoFile += argv[1];
  string outFile  = "./out/";  outFile  += argv[1];

  // Initialize the UAL parser
  UAL_SXF_Parser parser;

  // Read data
  parser.read(inFile.data(), echoFile.data());

  // Write data
  parser.write(outFile.data());

  return 1;
}
