#include "UAL/ADXF/elements/RfCavityWriter.hh"


UAL::ADXFRfCavityWriter::ADXFRfCavityWriter()
{
}

void UAL::ADXFRfCavityWriter::writeDesignElement(ostream& out, 
						 PacGenElement& element,
						 const std::string& tab)
{
  out << tab << "<rfcavity" 
      << " name=" << '\"' << element.name() << '\"';

  double l = getLength(element);
  
  if(l != 0.0) {
    out << " l="    << '\"' << getLength(element) << '\"';
  }
      
  out << " />" << std::endl; 
}

