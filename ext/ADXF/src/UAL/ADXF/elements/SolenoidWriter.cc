#include "UAL/ADXF/elements/SolenoidWriter.hh"


UAL::ADXFSolenoidWriter::ADXFSolenoidWriter()
{
}

void UAL::ADXFSolenoidWriter::writeDesignElement(ostream& out, 
						 PacGenElement& element,
						 const std::string& tab)
{
  out << tab << "<solenoid" 
      << " name=" << '\"' << element.name() << '\"';

  double l = getLength(element);
  
  if(l != 0.0) {
    out << " l="    << '\"' << getLength(element) << '\"';
  }
      
  out << " />" << std::endl; 
}

