#include "SMF/PacElemMultipole.h"


#include "UAL/ADXF/elements/QuadrupoleWriter.hh"


UAL::ADXFQuadrupoleWriter::ADXFQuadrupoleWriter()
{
}

void UAL::ADXFQuadrupoleWriter::writeDesignElement(ostream& out, PacGenElement& element,
						   const std::string& tab)
{
  out << tab << "<quadrupole" 
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

  writeDesignMultipole(out, element.name(), atts, tab);
    
  out << " />" << std::endl; 
}

void UAL::ADXFQuadrupoleWriter::writeDesignMultipole(ostream& out, 
						     const std::string& name, 
						     PacElemAttributes& atts,
						     const std::string& tab)
{
  PacElemAttributes::iterator it = atts.find(PAC_MULTIPOLE);
  if(it == atts.end()) return;

  double l = atts.get(PAC_L);
  if(l == 0) {
    cout << "warning: quadrupole " << name << ", length == 0 "  << std::endl;  
    return;
  }

  PacElemMultipole* mult = (PacElemMultipole*) &(*it);

  int norder = mult->size()/2;
  if(norder < 2) return;

  double k1  = mult->kl(1)/l;
  double kt1 = mult->ktl(1)/l;

  if(k1  != 0.0) out << " k1=" << '\"' << k1 << '\"';
  if(kt1 != 0.0) cout << "warning: adxf parser does not support " 
		      << "tilt in quadrupoles " << std::endl;  

}

