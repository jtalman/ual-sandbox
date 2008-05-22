#include "UAL/ADXF/elements/RcollimatorWriter.hh"


UAL::ADXFRcollimatorWriter::ADXFRcollimatorWriter()
{
}

void UAL::ADXFRcollimatorWriter::writeDesignElement(ostream& out, 
						    PacGenElement& element,
						    const std::string& tab)
{
  out << tab << "<rcollimator" 
      << " name=" << '\"' << element.name() << '\"';

  double l = getLength(element);
  
  if(l != 0.0) {
    out << " l="    << '\"' << getLength(element) << '\"';
  }
      
  out << " />" << std::endl; 
}

