#include "UAL/ADXF/elements/VmonitorWriter.hh"


UAL::ADXFVmonitorWriter::ADXFVmonitorWriter()
{
}

void UAL::ADXFVmonitorWriter::writeDesignElement(ostream& out, PacGenElement& element,
						 const std::string& tab)
{
  out << tab << "<vmonitor" 
      << " name=" << '\"' << element.name() << '\"';

  double l = getLength(element);
  
  if(l != 0.0) {
    out << " l="    << '\"' << getLength(element) << '\"';
  }
      
  out << " />" << std::endl; 
}

