#include <iomanip>
#include <stdio.h>
#include <stdlib.h>

#include "UAL/ADXF/Writer.hh"
#include "UAL/ADXF/ElementsWriter.hh"
#include "UAL/ADXF/SectorsWriter.hh"

extern FILE* yyin;
extern int yyparse();

// Constructor.
UAL::ADXFWriter::ADXFWriter()
{
  m_tab = "  ";
}


void UAL::ADXFWriter::write(const char* outFile)
{
  PacSmf smf;

  // Open an output file 
  std::ofstream out(outFile);
  if(!out) { 
    std::cerr << "Cannot open output file `" << outFile << "'." << std::endl; 
    exit(-1); 
  }

  out.precision(12);

  out << "<?xml version=" << '\"' << "1.0" << '\"' 
      <<  " encoding=" << '\"' << "utf-8" << '\"' << "?>" << std::endl;
  out << "<adxf xmlns" << ':' << "xsi=" << '\"' 
      << "http" << ':' << "//www.w3.org/2001/XMLSchema-instance" << '\"' << std::endl;
  out << "  xsi" << ':' << "noNamespaceSchemaLocation=" 
      << '\"' << "file" << ':' << "/home/xslt/ADXF/adxf.xsd" << '\"' << ">" << std::endl;

  // Write design elements 
  UAL::ADXFElementsWriter elementsWriter;
  elementsWriter.writeDesignElements(out, m_tab);
  // Write beamlines
  // accelerator.write(out);
  // Write real elements
  // accelerator.write(out);
  // Write sectors
  UAL::ADXFSectorsWriter sectorsWriter;
  sectorsWriter.writeSectors(out, m_tab);

  out << "</adxf>" << std::endl;

  out.close();
}
