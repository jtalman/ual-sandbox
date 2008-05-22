#include <iomanip>
#include <stdio.h>
#include <iostream>


#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLString.hpp>

#include "UAL/ADXF/Reader.hh"
#include "UAL/ADXF/ADXFHandler.hh"

extern FILE* yyin;
extern int yyparse();

XERCES_CPP_NAMESPACE_USE

UAL::ADXFReader* UAL::ADXFReader::s_theInstance = 0;

UAL::ADXFReader* UAL::ADXFReader::getInstance()
{
  if(s_theInstance == 0){
    try {
      XMLPlatformUtils::Initialize();
    }
    catch (const XMLException& toCatch) {
      char* message = XMLString::transcode(toCatch.getMessage());
      cout << "Error during initialization! :\n";
      cout << "Exception message is: \n"
	   << message << "\n";
      XMLString::release(&message);
      return 0;
    }
    s_theInstance = new UAL::ADXFReader();
  }
  return s_theInstance;
}

// Constructor.
UAL::ADXFReader::ADXFReader()
{
  m_tab = "  ";

  p_reader = XMLReaderFactory::createXMLReader();
  p_reader->setFeature(XMLUni::fgSAX2CoreValidation, true);   // optional
  p_reader->setFeature(XMLUni::fgSAX2CoreNameSpaces, true);   // optional
}

UAL::ADXFReader::~ADXFReader()
{
  if(p_reader) delete p_reader;
  s_theInstance = 0;
}


void UAL::ADXFReader::read(const char* adxfFile)
{

  // UAL::ADXFHandler* defaultHandler = new UAL::ADXFHandler(); // DefaultHandler();
  p_reader->setContentHandler(&m_docHandler);
  p_reader->setErrorHandler(&m_docHandler);

 try {
   p_reader->parse(adxfFile);
 }
 catch (const XMLException& toCatch) {
   char* message = XMLString::transcode(toCatch.getMessage());
   std::cout << "Exception message is: \n"
	<< message << "\n";
   XMLString::release(&message);
   exit(0);
 }
 catch (const SAXParseException& toCatch) {
   char* message = XMLString::transcode(toCatch.getMessage());
   std::cout << "Exception message is: \n"
	<< message << "\n";
   XMLString::release(&message);
   exit(0);
 }
 catch (...) {
   cout << "Unexpected Exception \n" ;
   exit(0);
 }

 return;

}
