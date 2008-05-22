#include "UAL/ADXF/elements/HmonitorWriter.hh"


UAL::ADXFHmonitorWriter::ADXFHmonitorWriter()
{
}

void UAL::ADXFHmonitorWriter::writeDesignElement(ostream& out, PacGenElement& element,
						 const std::string& tab)
{
  out << tab << "<hmonitor" 
      << " name=" << '\"' << element.name() << '\"';

  double l = getLength(element);
  
  if(l != 0.0) {
    out << " l="    << '\"' << getLength(element) << '\"';
  }
      
  out << " />" << std::endl; 
}

