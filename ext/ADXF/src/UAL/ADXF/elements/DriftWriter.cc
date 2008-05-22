#include "UAL/ADXF/elements/DriftWriter.hh"


UAL::ADXFDriftWriter::ADXFDriftWriter()
{
}

void UAL::ADXFDriftWriter::writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab)
{
  out << tab << "<drift" 
      << " name=" << '\"' << element.name() << '\"';

  double l = getLength(element);
  
  if(l != 0.0) {
    out << " l="    << '\"' << getLength(element) << '\"';
  }
      
  out << " />" << std::endl; 
}

