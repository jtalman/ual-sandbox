#include "SMF/PacElemMultipole.h"


#include "UAL/ADXF/elements/HkickerWriter.hh"


UAL::ADXFHkickerWriter::ADXFHkickerWriter()
{
}

void UAL::ADXFHkickerWriter::writeDesignElement(ostream& out, PacGenElement& element,
						   const std::string& tab)
{
  out << tab << "<hkicker" 
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

void UAL::ADXFHkickerWriter::writeDesignMultipole(ostream& out, 
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

  if(hkick  != 0.0) out << " kick=" << '\"' << hkick << '\"';
  if(vkick  != 0.0) cout << "warning: adxf parser does not support " 
		      << "tilt in hkicker " << std::endl;  
}

