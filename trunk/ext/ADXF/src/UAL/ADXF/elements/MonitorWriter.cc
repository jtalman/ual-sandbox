#include "UAL/ADXF/elements/MonitorWriter.hh"


UAL::ADXFMonitorWriter::ADXFMonitorWriter()
{
}

void UAL::ADXFMonitorWriter::writeDesignElement(ostream& out, PacGenElement& element,
						 const std::string& tab)
{
  out << tab << "<monitor" 
      << " name=" << '\"' << element.name() << '\"';

  double l = getLength(element);
  
  if(l != 0.0) {
    out << " l="    << '\"' << getLength(element) << '\"';
  }
      
  out << " />" << std::endl; 
}

