#include "UAL/ADXF/ElementWriter.hh"

UAL::ADXFElementWriter::ADXFElementWriter()
{
}


UAL::ADXFElementWriter::~ADXFElementWriter()
{
}

double UAL::ADXFElementWriter::getLength(PacGenElement& elem)
{
  PacElemPart* body = elem.getBody();;

  if(!body) { return 0.0; }

  return body->attributes().get(PAC_L);
}



