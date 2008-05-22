#include "SMF/PacElemMultipole.h"


#include "UAL/ADXF/elements/MultipoleWriter.hh"


UAL::ADXFMultipoleWriter::ADXFMultipoleWriter()
{
}

void UAL::ADXFMultipoleWriter::writeDesignElement(ostream& out, PacGenElement& element,
						  const std::string& tab)
{
  out << tab << "<multipole" 
      << " name=" << '\"' << element.name() << '\"';

  PacElemPart* body = element.getBody(); 

  if(!body) {
    out << " />" << std::endl; 
    return;
  }

  PacElemAttributes& atts = body->attributes();

  writeDesignMultipole(out, atts, tab);
    
  out << " />" << std::endl; 
}

void UAL::ADXFMultipoleWriter::writeDesignMultipole(ostream& out, 
						    PacElemAttributes& atts,
						    const std::string& tab)
{
  PacElemAttributes::iterator it = atts.find(PAC_MULTIPOLE);

  if(it == atts.end()) return;

  PacElemMultipole* mult = (PacElemMultipole*) &(*it);

  int norder = mult->size()/2;

  int f = 1;
  double kl, ktl;
  for(int io = 0; io < norder; io++){
    if(io > 0) f *= io;
    kl = mult->kl(io);
    ktl = mult->ktl(io);
    if(kl != 0.0) out << " k" << io << "l=" << '\"' << kl/f << '\"';
    if(ktl != 0.0) cout << "warning: adxf parser does not support " 
			<< "roll angle in thin multipoles " << std::endl;  
  }
}

