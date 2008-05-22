#include "SMF/PacElemMultipole.h"


#include "UAL/ADXF/elements/SextupoleWriter.hh"


UAL::ADXFSextupoleWriter::ADXFSextupoleWriter()
{
}

void UAL::ADXFSextupoleWriter::writeDesignElement(ostream& out, PacGenElement& element,
						   const std::string& tab)
{
  out << tab << "<sextupole" 
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

void UAL::ADXFSextupoleWriter::writeDesignMultipole(ostream& out, 
						    PacElemAttributes& atts,
						    const std::string& tab)
{
  PacElemAttributes::iterator it = atts.find(PAC_MULTIPOLE);
  if(it == atts.end()) return;

  double l = atts.get(PAC_L);
  if(l == 0) {
    cout << "warning: sextupole length == 0 "  << std::endl;  
    return;
  }

  PacElemMultipole* mult = (PacElemMultipole*) &(*it);

  int norder = mult->size()/2;
  if(norder < 3) return;

  double k2  = mult->kl(2)/l/2;
  double kt2 = mult->ktl(2)/l/2;

  if(k2  != 0.0) out << " k2=" << '\"' << k2 << '\"';
  if(kt2 != 0.0) cout << "warning: adxf parser does not support " 
		      << "tilt in sextupoles " << std::endl;  

}

