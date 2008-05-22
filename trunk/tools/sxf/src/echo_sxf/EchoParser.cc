#include <iomanip>
#include <stdio.h>
#include <fstream>
#include "echo_sxf/EchoParser.hh"
#include "echo_sxf/EchoAcceleratorReader.hh"

extern FILE* yyin;
extern int yyparse();

// Constructor.
SXF::EchoParser::EchoParser()
{
}

// Analyze input file and write its echo (and error messages). 
void SXF::EchoParser::read(const char* inFile, const char* eFile)
{
 
  // Open an input file.
  yyin = fopen(inFile, "r");

  // Open an echo file. 
  ofstream echoFile(eFile);

  // Make SXF_OStream.
  SXF::OStream echo(echoFile);

  // Initialize a parser.
  SXF::AcceleratorReader::s_reader = new SXF::EchoAcceleratorReader(echo);

  // Parse an input stream.
  yyparse();

  // Release data.
  delete SXF::AcceleratorReader::s_reader; SXF::AcceleratorReader::s_reader = 0;

  // Close an error file
  echoFile.close();
  
}

