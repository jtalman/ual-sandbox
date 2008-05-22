#include "SMF/PacElemMultipole.h"


#include "UAL/ADXF/elements/VkickerWriter.hh"


UAL::ADXFVkickerWriter::ADXFVkickerWriter()
{
}

void UAL::ADXFVkickerWriter::writeDesignElement(ostream& out, PacGenElement& element,
						   const std::string& tab)
{
  out << tab << "<vkicker" 
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

  writeDesignMultipole(out, atts, tab);
    
  out << " />" << std::endl; 
}

void UAL::ADXFVkickerWriter::writeDesignMultipole(ostream& out, 
						    PacElemAttributes& atts,
						    const std::string& tab)
{
  PacElemAttributes::iterator it = atts.find(PAC_MULTIPOLE);
  if(it == atts.end()) return;

  PacElemMultipole* mult = (PacElemMultipole*) &(*it);

  int norder = mult->size()/2;
  if(norder < 1) return;

  double hkick  = mult->kl(0);
  double vkick  = mult->ktl(0);

  if(vkick  != 0.0) out << " kick=" << '\"' << vkick << '\"';
  if(hkick  != 0.0) cout << "warning: adxf parser does not support " 
		      << "tilt in vkicker " << std::endl;  
}

