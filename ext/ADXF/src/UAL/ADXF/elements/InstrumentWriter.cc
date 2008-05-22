#include "UAL/ADXF/elements/InstrumentWriter.hh"


UAL::ADXFInstrumentWriter::ADXFInstrumentWriter()
{
}

void UAL::ADXFInstrumentWriter::writeDesignElement(ostream& out, PacGenElement& element,
						   const std::string& tab)
{
  out << tab << "<instrument" 
      << " name=" << '\"' << element.name() << '\"';

  double l = getLength(element);
  
  if(l != 0.0) {
    out << " l="    << '\"' << getLength(element) << '\"';
  }
      
  out << " />" << std::endl; 
}

