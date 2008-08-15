
#include <iostream>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/ADXFHandler.hh"
#include "UAL/ADXF/ConstantsHandler.hh"
#include "UAL/ADXF/ElementsHandler.hh"
#include "UAL/ADXF/SectorsHandler.hh"

XERCES_CPP_NAMESPACE_USE

UAL::ADXFBasicHandler::ADXFBasicHandler()
{
}

UAL::ADXFBasicHandler::~ADXFBasicHandler()
{
}


void UAL::ADXFBasicHandler::startElement(const   XMLCh* const    uri,
					 const   XMLCh* const    localname,
					 const   XMLCh* const    qname,
					 const   xercesc::Attributes&     attrs)
{

}

void UAL::ADXFBasicHandler::endElement(const   XMLCh* const    uri,
				       const   XMLCh* const    localname,
				       const   XMLCh* const    qname)
{
}

void UAL::ADXFBasicHandler::fatalError(const SAXParseException& exception)
{
    char* message = XMLString::transcode(exception.getMessage());
    cout << "Fatal Error: " << message
         << " at line: " << exception.getLineNumber()
         << endl;
}

