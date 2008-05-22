#include "SMF/PacElemBend.h"

#include "UAL/ADXF/elements/SbendWriter.hh"


UAL::ADXFSbendWriter::ADXFSbendWriter()
{
}

void UAL::ADXFSbendWriter::writeDesignElement(ostream& out, PacGenElement& element,
					      const std::string& tab)
{
  out << tab << "<sbend" 
      << " name=" << '\"' << element.name() << '\"';

  PacElemPart* body = element.getBody(); 

  if(!body) {
      out << " />" << std::endl; 
      return;
  }

  double l = getLength(element);
  
  if(l != 0.0) {
    out << " l="    << '\"' << getLength(element) << '\"';
  }

  PacElemAttributes& atts = body->attributes();

  writeDesignBend(out, atts, tab);
      
  out << " />" << std::endl; 
}

void UAL::ADXFSbendWriter::writeDesignBend(ostream& out, 
					   PacElemAttributes& atts,
					   const std::string& tab)
{
  PacElemAttributes::iterator it = atts.find(PAC_BEND);

  if(it == atts.end()) return;

  PacElemBend* bend = (PacElemBend*) &(*it);

  double angle = bend->angle();

  if(angle != 0.0) out << " angle=" << '\"' << angle << '\"';
}

