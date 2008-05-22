#include <iomanip>
#include <stdio.h>
#include "ual_sxf/Parser.hh"
#include "ual_sxf/AcceleratorReader.hh"

extern FILE* yyin;
extern int yyparse();

// Constructor.
UAL_SXF_Parser::UAL_SXF_Parser()
{
}

void UAL_SXF_Parser::read(const char* inFile, const char* eFile)
{
  PacSmf smf;

  // Open an input file
  yyin = fopen(inFile, "r");

  // Open an echo file 
  ofstream echoFile(eFile);

  // Make SXF::OStream
  SXF::OStream echo(echoFile);

  // Initialize a parser
  SXF::AcceleratorReader::s_reader = new UAL_SXF_AcceleratorReader(echo, smf);

  // Parse an input stream
  yyparse();

  // Release data
  delete SXF::AcceleratorReader::s_reader; SXF::AcceleratorReader::s_reader = 0;

  // Close an error file
  echoFile.close();
  
}

void UAL_SXF_Parser::write(const char* outFile)
{
  PacSmf smf;

  // Open an output file 
  ofstream out(outFile);
  if(!out) { 
    cerr << "Cannot open output file `"<<outFile<<"'.\n"; 
    exit(-1); 
  }

  out.precision(12);

  // Open an error stream
  SXF::OStream error_stream(out);

  // Prepare an accelerator parser  
  UAL_SXF_AcceleratorReader* accelerator = 
    new UAL_SXF_AcceleratorReader(error_stream, smf);
  // Write data
  accelerator->write(out);

  // Release data
  delete accelerator;

  out.close();
}
