#include "UAL/ADXF/elements/MarkerWriter.hh"


UAL::ADXFMarkerWriter::ADXFMarkerWriter()
{
}

void UAL::ADXFMarkerWriter::writeDesignElement(ostream& out, PacGenElement& element,
				    const std::string& tab)
{
  out << tab << "<marker" 
      << " name=" << '\"' << element.name() << '\"' 
      << " />" << std::endl; 
}

